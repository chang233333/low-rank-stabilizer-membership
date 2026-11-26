"""Exact certificates for rank-parameterized qubit stabilizer membership.

The default implementation uses a rational phase-I simplex with Bland's rule,
and verifies each primal or dual certificate.  The theoretical polynomial LP
bound uses a polynomial-time rational LP algorithm, not this simplex routine.
An optional HiGHS proposal mode also checks its output exactly.  This is not a
numerical weak-membership solver and does not compute robustness of magic.
"""
from itertools import product
from math import lcm
from fractions import Fraction
import time

import numpy as np
import sympy as sp
from scipy.optimize import linprog

I = sp.I
ROOTS = (sp.Integer(1), I, sp.Integer(-1), -I)
PHASE_R = np.array([1, 0, -1, 0], dtype=np.int64)
PHASE_I = np.array([0, 1, 0, -1], dtype=np.int64)


def phase_vector(real, imag):
    """Projective phase encoding, or None; all operations are integer exact."""
    real, imag = np.asarray(real), np.asarray(imag)
    nz = np.flatnonzero((real != 0) | (imag != 0))
    if not len(nz):
        return None
    a, b = int(real[nz[0]]), int(imag[nz[0]])
    rotations = [(a, b), (-b, a), (-a, -b), (b, -a)]
    result = np.full(len(real), -1, dtype=np.int8)
    for k, (x, y) in enumerate(rotations):
        result[(real == x) & (imag == y)] = k
    if np.any(result[nz] < 0):
        return None
    return result


def stabilizer_certificate(phases):
    """Check the affine-support/quadratic-phase normal form for qubits."""
    if phases is None:
        return None
    d = len(phases)
    if d <= 0 or d & (d - 1):
        raise ValueError("Dimension must be a power of two")
    support = np.flatnonzero(phases >= 0).tolist()
    count = len(support)
    if not count or count & (count - 1):
        return None
    offset = support[0]
    pivots = {}
    for x in support:
        v = x ^ offset
        while v:
            p = v.bit_length() - 1
            if p in pivots:
                v ^= pivots[p]
            else:
                pivots[p] = v
                break
    basis = list(pivots.values())
    k = len(basis)
    if count != 1 << k:
        return None
    locations = []
    for t in range(1 << k):
        x = offset
        for j, b in enumerate(basis):
            if (t >> j) & 1:
                x ^= b
        locations.append(x)
    if any(phases[x] < 0 for x in locations):
        return None
    q = [int(phases[x]) for x in locations]
    base = q[0]
    q = [(z - base) % 4 for z in q]
    linear = [q[1 << j] for j in range(k)]
    quadratic = []
    for j in range(k):
        for h in range(j + 1, k):
            coeff = (q[(1 << j) | (1 << h)] - linear[j] - linear[h]) % 4
            if coeff not in (0, 2):
                return None
            if coeff:
                quadratic.append((j, h))
    for t, actual in enumerate(q):
        expect = sum(linear[j] for j in range(k) if (t >> j) & 1)
        expect += 2 * sum(((t >> j) & 1) * ((t >> h) & 1) for j, h in quadratic)
        if expect % 4 != actual:
            return None
    return {"affine_offset": offset, "basis_xor": basis,
            "linear_mod4": linear, "quadratic_mod2_pairs": quadratic,
            "support_size": count}


def enumerate_in_span(columns, keep_phases=True):
    """Enumerate every stabilizer ray in the given exact Q(i) column span."""
    start = time.perf_counter()
    x = sp.Matrix(columns)
    independent = list(x.rref()[1])
    base = x[:, independent]
    r = base.cols
    if not r:
        return {"rank": 0, "dimension": x.rows, "candidates": []}
    rows = list(base.T.rref()[1])
    bmat = (base * base[rows, :].inv()).applyfunc(sp.simplify)
    assert bmat[rows, :] == sp.eye(r)
    re = [sp.re(z).expand(complex=True) for z in bmat]
    im = [sp.im(z).expand(complex=True) for z in bmat]
    if not all(z.is_Rational for z in re + im):
        raise ValueError("Prototype input field is Q(i)")
    denominator = lcm(*(int(z.q) for z in re + im))
    re_int = [int(z * denominator) for z in re]
    im_int = [int(z * denominator) for z in im]
    max_entry = max([abs(z) for z in re_int + im_int] + [1])
    dtype = np.int64 if max_entry * max(1, 2 * r) < 2**60 else object
    br = np.array(re_int, dtype=dtype).reshape(bmat.shape)
    bi = np.array(im_int, dtype=dtype).reshape(bmat.shape)
    candidates = []
    checked = 0
    # First nonzero pivot amplitude is fixed to 1, quotienting global phase.
    for first in range(r):
        for tail in product(range(-1, 4), repeat=r-first-1):
            pattern = [-1] * first + [0] + list(tail)
            ar = np.array([0 if z < 0 else PHASE_R[z] for z in pattern], dtype=dtype)
            ai = np.array([0 if z < 0 else PHASE_I[z] for z in pattern], dtype=dtype)
            phases = phase_vector(br @ ar - bi @ ai, br @ ai + bi @ ar)
            checked += 1
            cert = stabilizer_certificate(phases)
            if cert is not None:
                alpha = sp.Matrix([0 if z < 0 else ROOTS[z] for z in pattern])
                item={"alpha":alpha,"normal_form":cert}
                if keep_phases:item['phase_encoding']=phases
                candidates.append(item)
    assert checked == (5**r - 1) // 4
    keys = [tuple(c["alpha"]) for c in candidates]
    assert len(set(keys)) == len(keys)
    return {"dimension": x.rows, "rank": r, "pivot_rows": rows,
            "basis": bmat, "candidate_patterns_checked": checked,
            "candidates": candidates, "elapsed_seconds": time.perf_counter()-start}


def hermitian_coordinates(a):
    a = sp.Matrix(a)
    r = a.rows
    assert a == a.conjugate().T
    out = [sp.re(a[j, j]).expand(complex=True) for j in range(r)]
    for j in range(r):
        for k in range(j+1, r):
            out += [sp.re(a[j, k]).expand(complex=True),
                    sp.im(a[j, k]).expand(complex=True)]
    assert len(out) == r*r
    return sp.Matrix(out)


def exact_conic_feasibility(a, h):
    """A finite rational phase-I simplex; return a checked primal or dual.

    Bland's entering/leaving rules prevent cycling.  This implementation is
    not claimed to attain the polynomial-time LP complexity used in the theorem.
    """
    a, h = sp.Matrix(a), sp.Matrix(h)
    row_ids = list(a.T.rref()[1])
    rank = len(row_ids)
    if a.row_join(h).rank() > rank:
        for z in a.T.nullspace():
            value = (h.T*z)[0]
            if value != 0:
                z = -z if value > 0 else z
                assert a.T*z == sp.zeros(a.cols,1) and (h.T*z)[0] < 0
                return False, z, 0
        raise AssertionError("Missing left-nullspace certificate")
    if rank == 0:
        assert h == sp.zeros(h.rows,1)
        return True, sp.zeros(a.cols,1), 0
    reduced = a[row_ids,:]
    rhs = h[row_ids,:]
    signs = [1 if rhs[j] >= 0 else -1 for j in range(rank)]
    signed = sp.diag(*signs)
    reduced, rhs = signed*reduced, signed*rhs
    tableau = reduced.row_join(sp.eye(rank))
    transform = sp.eye(rank)
    basis = list(range(a.cols,a.cols+rank))
    costs = [sp.Integer(0)]*a.cols + [sp.Integer(1)]*rank
    iterations = 0
    while True:
        c_basic = sp.Matrix([[costs[j] for j in basis]])
        reduced_costs = sp.Matrix([costs]) - c_basic*tableau
        nonbasic = set(range(a.cols+rank))-set(basis)
        entering = next((j for j in sorted(nonbasic) if reduced_costs[j] < 0),None)
        if entering is None:
            break
        eligible = [i for i in range(rank) if tableau[i,entering] > 0]
        assert eligible, "Phase-I objective cannot be unbounded below"
        ratio = min(rhs[i]/tableau[i,entering] for i in eligible)
        ties = [i for i in eligible if rhs[i]/tableau[i,entering] == ratio]
        leaving = min(ties,key=lambda i:basis[i])
        pivot = tableau[leaving,entering]
        tableau[leaving,:] = tableau[leaving,:]/pivot
        transform[leaving,:] = transform[leaving,:]/pivot
        rhs[leaving] /= pivot
        for i in range(rank):
            if i == leaving:
                continue
            factor = tableau[i,entering]
            if factor:
                tableau[i,:] = tableau[i,:]-factor*tableau[leaving,:]
                transform[i,:] = transform[i,:]-factor*transform[leaving,:]
                rhs[i] -= factor*rhs[leaving]
        basis[leaving] = entering
        iterations += 1
    objective = (sp.Matrix([[costs[j] for j in basis]])*rhs)[0]
    assert objective >= 0
    if objective == 0:
        x = sp.zeros(a.cols,1)
        for i,j in enumerate(basis):
            if j < a.cols:
                x[j] = rhs[i]
        assert all(v >= 0 for v in x) and a*x == h
        return True,x,iterations
    pi = (sp.Matrix([[costs[j] for j in basis]])*transform).T
    z = sp.zeros(a.rows,1)
    for i,j in enumerate(row_ids):
        z[j] = -signs[i]*pi[i]
    assert all(v >= 0 for v in a.T*z) and (h.T*z)[0] < 0
    return False,z,iterations


def compressed_membership(enumeration, pivot_density, exact=True):
    """Return a rationally verified compressed YES or NO certificate."""
    candidates = enumeration["candidates"]
    h = hermitian_coordinates(pivot_density)
    r = enumeration["rank"]
    a = sp.Matrix.hstack(*[hermitian_coordinates(c["alpha"]*c["alpha"].conjugate().T)
                          for c in candidates]) if candidates else sp.zeros(r*r, 0)
    assert all(z in (-1, 0, 1) for z in a)
    if not candidates:
        return {"is_free": False, "certificate": "no stabilizer ray in support",
                "equalities": r*r, "columns": 0}
    if exact:
        feasible, certificate, pivots = exact_conic_feasibility(a,h)
        if feasible:
            weights = certificate
            active = [j for j,v in enumerate(weights) if v > 0]
            return {"is_free": True, "certificate": "rational phase-I simplex and exact nonnegative decomposition",
                    "equalities": r*r, "columns": a.cols,
                    "simplex_pivots": pivots,
                    "active_candidates": active,
                    "conic_weights": [str(weights[j]) for j in active]}
        else:
            z = certificate
            return {"is_free": False, "certificate": "rational phase-I simplex and exact compressed Farkas separator",
                    "equalities": r*r, "columns": a.cols,
                    "simplex_pivots": pivots,
                    "dual_coordinates": [str(v) for v in z],
                    "dual_target": str((h.T*z)[0])}
    af, hf = np.array(a, dtype=float), np.array(h, dtype=float).ravel()
    result = linprog(np.ones(a.cols), A_eq=af, b_eq=hf,
                     bounds=(0, None), method="highs")
    if result.success:
        active = np.flatnonzero(result.x > 1e-9).tolist()
        solution, params = a[:, active].gauss_jordan_solve(h)
        if params.rows:
            solution = solution.subs({x: 0 for x in params})
        assert all(z >= 0 for z in solution)
        assert a[:, active] * solution == h
        return {"is_free": True, "certificate": "exact nonnegative decomposition",
                "equalities": r*r, "columns": a.cols,
                "active_candidates": active,
                "conic_weights": [str(z) for z in solution]}
    if result.status != 2:
        raise RuntimeError(f"LP undecided: {result.message}")
    dual = linprog(hf, A_ub=-af.T, b_ub=np.zeros(a.cols),
                   bounds=[(-1, 1)]*(r*r), method="highs")
    assert dual.success and dual.fun < -1e-10
    z = sp.Matrix([sp.Rational(Fraction(float(x)).limit_denominator(10**8)) for x in dual.x])
    assert all(v >= 0 for v in a.T*z)
    assert (h.T*z)[0] < 0
    return {"is_free": False, "certificate": "exact compressed Farkas separator",
            "equalities": r*r, "columns": a.cols,
            "dual_coordinates": [str(v) for v in z],
            "dual_target": str((h.T*z)[0])}


def vector_from_phases(phases):
    return sp.Matrix([0 if z < 0 else ROOTS[int(z)] for z in phases])


def global_magic_witness(enumeration, pivot_density, decision):
    """Lift a compressed NO certificate to a global stabilizer witness.

    Returns an exact factorized witness W=R^* Z R+kappa*(I-C G^-1 C^*).
    Its sign convention is nonnegative on all stabilizer states and negative
    on the target.  A conservative depolarizing-noise threshold is included.
    """
    if decision['is_free']:
        raise ValueError('A NO decision is required')
    b=enumeration['basis'];d,r=b.shape
    h=hermitian_coordinates(pivot_density)
    coords=sp.Matrix([sp.Rational(x) for x in decision['dual_coordinates']]) if 'dual_coordinates' in decision else -h
    assert (h.T*coords)[0] < 0
    z=sp.zeros(r,r)
    pos=0
    for j in range(r):z[j,j]=coords[pos];pos+=1
    for j in range(r):
        for k in range(j+1,r):
            z[j,k]=(coords[pos]+I*coords[pos+1])/2
            z[k,j]=sp.conjugate(z[j,k]);pos+=2
    den=lcm(*(int(sp.re(x).q) for x in b),*(int(sp.im(x).q) for x in b))
    c=b*den
    gram=(c.conjugate().T*c).applyfunc(sp.simplify)
    q=sp.simplify(gram.det())
    assert q.is_Integer and q>0
    norm_bound=sum(abs(x) for x in coords)
    kappa=q*d*norm_bound
    target=(h.T*coords)[0]
    white=(sp.trace(z)+kappa*(d-r))/d
    assert white>=0
    threshold=-target/(white-target)
    return {'pivot_rows':enumeration['pivot_rows'],'compressed_witness':z,
            'integer_basis':c,'gram_inverse':gram.inv(),'integer_gap_denominator':q,
            'norm_bound':norm_bound,'penalty':kappa,'target_expectation':target,
            'white_noise_expectation':white,'certified_depolarizing_threshold':threshold}


def materialize_witness(witness):
    c=witness['integer_basis'];d,r=c.shape
    selector=sp.zeros(r,d)
    for i,j in enumerate(witness['pivot_rows']):selector[i,j]=1
    return (selector.T*witness['compressed_witness']*selector+
            witness['penalty']*(sp.eye(d)-c*witness['gram_inverse']*c.conjugate().T)).applyfunc(sp.simplify)


def stabilizer_membership(rho):
    """Exact qubit membership from an explicit Hermitian trace-one Q(i) matrix.

    Returns (support_enumeration, certified_decision).  The support enumeration
    contains SymPy matrices and therefore is not directly JSON serializable.
    Rank is exact: this API does not threshold small eigenvalues.
    Positivity need not be promised: a YES decomposition guarantees it, and a
    Hermitian trace-one input outside the positive cone is correctly rejected.
    """
    rho=sp.Matrix(rho)
    if rho.rows != rho.cols or rho.rows < 2 or rho.rows & (rho.rows-1):
        raise ValueError("Input must be a 2^n by 2^n matrix, n>=1")
    entries=[]
    for z in rho:
        real=sp.re(z).expand(complex=True)
        imag=sp.im(z).expand(complex=True)
        if not real.is_Rational or not imag.is_Rational:
            raise ValueError("Prototype requires exact rational real and imaginary parts")
        entries.append(real+I*imag)
    # Canonical Gaussian rationals avoid rejecting algebraically equal inputs
    # merely because SymPy has retained products such as (1-I)*(1+I).
    rho=sp.Matrix(rho.rows,rho.cols,entries)
    if rho != rho.conjugate().T or sp.trace(rho) != 1:
        raise ValueError("Input must be Hermitian with exact trace one")
    enumeration=enumerate_in_span(rho,keep_phases=False)
    rows=enumeration['pivot_rows']
    decision=compressed_membership(enumeration,rho[rows,rows])
    return enumeration,decision
