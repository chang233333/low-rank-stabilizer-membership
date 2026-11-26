"""Exact stabilizer-ray search using injective Pauli postselection."""

from itertools import combinations, product
from math import prod

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed

from stabilizer_lp import (
    ROOTS,
    compressed_membership,
    enumerate_in_span,
    stabilizer_certificate,
)


FIELD = sp.QQ_I
PHASES = tuple(FIELD.from_sympy(z) for z in ROOTS)
PHASE_INDEX = {FIELD.zero: -1, **{z: j for j, z in enumerate(PHASES)}}


def _domain_matrix(matrix):
    matrix = sp.Matrix(matrix)
    if matrix.has(sp.Float):
        raise ValueError("Expected exact Gaussian rational entries")
    try:
        return matrix.to_DM(domain=FIELD).to_dense()
    except CoercionFailed as error:
        raise ValueError("Expected exact Gaussian rational entries") from error


def _parity(mask):
    return mask.bit_count() % 2


def _postselect_rows(rows, shift, phase_mask, outcome):
    """Return a scaled Clifford decoding of one Pauli eigenspace."""
    if not shift:
        return [row for x, row in enumerate(rows) if _parity(x & phase_mask) == outcome]

    pivot_bit = shift & -shift
    phase = _parity(shift & phase_mask)
    result = []
    for x, row in enumerate(rows):
        if x & pivot_bit:
            continue
        partner = x ^ shift
        exponent = phase + 2 * (outcome ^ _parity(partner & phase_mask))
        weight = PHASES[exponent]
        result.append([a + weight * b for a, b in zip(row, rows[partner])])
    return result


def _compress(basis):
    rank = basis.shape[1]
    measurements = []
    while basis.shape[0] >= 2 * rank:
        dimension = basis.shape[0]
        rows = basis.to_list()
        selection = None
        for shift, phase_mask in product(range(dimension), repeat=2):
            if not (shift or phase_mask):
                continue
            for outcome in (0, 1):
                projected = DomainMatrix(
                    _postselect_rows(rows, shift, phase_mask, outcome),
                    (dimension // 2, rank),
                    FIELD,
                )
                if projected.rank() == rank:
                    selection = projected
                    measurements.append((dimension, shift, phase_mask, outcome))
                    break
            if selection is not None:
                break
        if selection is None:
            break
        basis = selection
    return basis, measurements


def _binary_span(basis):
    span = [0]
    for row in basis:
        span += [row ^ x for x in span]
    return span


def _stabilizer_rays(dimension):
    """Yield one Gaussian integer vector for each stabilizer ray."""
    qubits = dimension.bit_length() - 1
    for size in range(qubits + 1):
        pairs = tuple(combinations(range(size), 2))
        for pivots in combinations(range(qubits), size):
            free = [j for j in range(qubits) if j not in pivots]
            slots = [(i, j) for i, pivot in enumerate(pivots) for j in free if j > pivot]
            for fill in product((0, 1), repeat=len(slots)):
                basis = [1 << j for j in pivots]
                for (i, j), bit in zip(slots, fill):
                    basis[i] |= bit << j
                span = _binary_span(basis)
                for offset in _binary_span([1 << j for j in free]):
                    support = [offset ^ x for x in span]
                    for linear in product(range(4), repeat=size):
                        for quadratic in product((0, 2), repeat=len(pairs)):
                            exponents = [0] * (1 << size)
                            for i, coeff in enumerate(linear):
                                for mask in range(len(exponents)):
                                    if mask & (1 << i):
                                        exponents[mask] += coeff
                            for (i, j), coeff in zip(pairs, quadratic):
                                bits = (1 << i) | (1 << j)
                                for mask in range(len(exponents)):
                                    if mask & bits == bits:
                                        exponents[mask] += coeff
                            vector = [[FIELD.zero] for _ in range(dimension)]
                            for point, exponent in zip(support, exponents):
                                vector[point] = [PHASES[exponent % 4]]
                            yield DomainMatrix(vector, (dimension, 1), FIELD)


def _phase_encoding(vector):
    entries = [row[0] for row in vector.to_list()]
    scale = next(z for z in entries if z)
    phases = [PHASE_INDEX.get(z / scale) for z in entries]
    if None in phases:
        return None
    return np.asarray(phases, dtype=np.int8)


def enumerate_compressed(columns, *, keep_phases=True):
    """Enumerate stabilizer rays in the exact column space of columns."""
    matrix = _domain_matrix(columns)
    dimension = matrix.shape[0]
    if not dimension or dimension & (dimension - 1):
        raise ValueError("The row dimension must be a power of two")
    _, independent = matrix.rref()
    rank = len(independent)
    if not rank:
        return {"dimension": dimension, "rank": 0, "candidates": []}

    basis = matrix.extract(range(dimension), independent)
    _, pivot_rows = basis.transpose().rref()
    basis = basis.matmul(basis.extract(pivot_rows, range(rank)).inv())
    reduced, measurements = _compress(basis)
    reduced_dimension = reduced.shape[0]
    _, reduced_pivots = reduced.transpose().rref()
    pivot_inverse = reduced.extract(reduced_pivots, range(rank)).inv()

    alphabet_count = (5**rank - 1) // 4
    qubits = reduced_dimension.bit_length() - 1
    stabilizer_count = reduced_dimension * prod((1 << j) + 1 for j in range(1, qubits + 1))
    if alphabet_count < stabilizer_count:
        small = enumerate_in_span(reduced.to_Matrix())
        rays = (
            DomainMatrix(
                [[FIELD.zero if j < 0 else PHASES[int(j)]] for j in item["phase_encoding"]],
                (reduced_dimension, 1),
                FIELD,
            )
            for item in small["candidates"]
        )
        pattern_count = alphabet_count
    else:
        rays = _stabilizer_rays(reduced_dimension)
        pattern_count = stabilizer_count

    candidates = []
    considered_lifts = 0
    for ray in rays:
        coordinates = pivot_inverse.matmul(ray.extract(reduced_pivots, [0]))
        if reduced.matmul(coordinates) != ray:
            continue
        considered_lifts += 1
        phases = _phase_encoding(basis.matmul(coordinates))
        certificate = stabilizer_certificate(phases)
        if certificate is None:
            continue
        pivot_phases = [int(phases[i]) for i in pivot_rows]
        pivot_phase = next(j for j in pivot_phases if j >= 0)
        alpha = sp.Matrix([0 if j < 0 else ROOTS[(j - pivot_phase) % 4] for j in pivot_phases])
        item = {"alpha": alpha, "normal_form": certificate}
        if keep_phases:
            item["phase_encoding"] = phases
        candidates.append(item)

    return {
        "dimension": dimension,
        "rank": rank,
        "pivot_rows": list(pivot_rows),
        "basis": basis.to_Matrix(),
        "compressed_dimension": reduced_dimension,
        "postselections": measurements,
        "candidate_patterns_checked": pattern_count,
        "considered_lifts": considered_lifts,
        "candidates": candidates,
    }


def stabilizer_membership(rho):
    """Decide exact stabilizer membership and return the compressed LP certificate."""
    rho = _domain_matrix(rho).to_Matrix()
    if rho.rows != rho.cols or not rho.rows or rho.rows & (rho.rows - 1):
        raise ValueError("Expected a square density matrix of power-of-two dimension")
    if rho != rho.H or sp.trace(rho) != 1:
        raise ValueError("Expected a Hermitian matrix of trace one")

    enumeration = enumerate_compressed(rho, keep_phases=False)
    rows = enumeration["pivot_rows"]
    return enumeration, compressed_membership(enumeration, rho[rows, rows])
