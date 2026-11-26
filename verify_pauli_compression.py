"""Check the Pauli compression against exact stabilizer and LP certificates."""

from itertools import product

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from pauli_compression import (
    FIELD,
    _phase_encoding,
    _postselect_rows,
    _stabilizer_rays,
    enumerate_compressed,
    stabilizer_membership,
)
from stabilizer_lp import (
    enumerate_in_span,
    global_magic_witness,
    materialize_witness,
    stabilizer_certificate,
)


def rays_in(enumeration):
    return {tuple(candidate["alpha"]) for candidate in enumeration["candidates"]}


def verify_postselection():
    rays = list(_stabilizer_rays(4))
    assert len({tuple(ray.flat()) for ray in rays}) == 60 == len(rays)
    count = 0
    for ray, (shift, mask), outcome in product(rays, product(range(4), repeat=2), (0, 1)):
        if not (shift or mask):
            continue
        projected = DomainMatrix(_postselect_rows(ray.to_list(), shift, mask, outcome), (2, 1), FIELD)
        if not projected.is_zero_matrix:
            assert stabilizer_certificate(_phase_encoding(projected)) is not None
        count += 1
    print(f"{count} two-qubit stabilizer postselections checked")


def verify_enumeration():
    eye = sp.eye(8)
    cases = [
        eye[:, [0, 1, 2, 4]],
        sp.eye(16)[:, [0, 1, 2, 3, 4, 8, 12]],
        sp.Matrix.hstack(eye[:, 0] + 2 * eye[:, 1], eye[:, 2] + sp.I * eye[:, 7]),
    ]
    results = []
    for columns in cases:
        result = enumerate_compressed(columns)
        baseline = enumerate_in_span(columns)
        assert rays_in(result) == rays_in(baseline)
        results.append(result)
    assert results[0]["postselections"][0][1] == 7
    assert not results[1]["postselections"]
    assert len(results[2]["candidates"]) == 1

    code = enumerate_compressed(sp.eye(64)[:, :8])
    assert code["compressed_dimension"] == 8
    assert len(code["candidates"]) == code["candidate_patterns_checked"] == 1080
    print("3 exact span comparisons; rank-8 code uses 1,080 candidates out of 97,656 alphabet patterns")


def verify_lp_certificates():
    eye = sp.eye(4)
    magic = eye[:, 1] + 2 * eye[:, 2]
    rho = (eye[:, 0] * eye[:, 0].H + magic * magic.H) / 6
    enum, decision = stabilizer_membership(rho)
    assert not decision["is_free"] and len(enum["candidates"]) == 1
    rows = enum["pivot_rows"]
    witness = materialize_witness(global_magic_witness(enum, rho[rows, rows], decision))
    assert sp.trace(witness * rho) == -sp.Rational(1, 6)
    for ray in _stabilizer_rays(4):
        v = ray.to_Matrix()
        assert sp.simplify((v.H * witness * v)[0]) >= 0

    rho = sp.diag(sp.Rational(1, 3), sp.Rational(2, 3), 0, 0)
    enum, decision = stabilizer_membership(rho)
    assert decision["is_free"]
    reconstructed = sp.zeros(rho.rows)
    for j, weight in zip(decision["active_candidates"], decision["conic_weights"]):
        v = enum["basis"] * enum["candidates"][j]["alpha"]
        reconstructed += sp.Rational(weight) * v * v.H
    assert reconstructed == rho
    print("Exact NO witness and YES decomposition checked")


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without Python's -O flag to enable exact assertions")
    verify_postselection()
    verify_enumeration()
    verify_lp_certificates()
