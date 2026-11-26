# Exact Stabilizer Membership for Low-Rank Density Matrices

Exact algorithms for membership in the stabilizer polytope, parameterized by the ordinary matrix rank of a quantum state. The method uses injective Pauli postselection to compress the support before enumerating stabilizer rays.

**[Read the manuscript (PDF)](paper/main.pdf)** | [LaTeX source](paper/main.tex) | [Python implementation](pauli_compression.py) | [Lean proofs](lean/QuantumOpen/PauliCompression.lean)

## Main results

For an explicitly encoded rational qubit density matrix of dimension $D$, rank $r$, and input bit length $L$:

- Exact deterministic membership testing in $2^{O(\log^2(r+1))}\operatorname{poly}(D,L)$ bit operations.
- Injective Pauli compression to a space of dimension at most $r^2$, preserving stabilizer rays in the support.
- For a member, a rational convex decomposition with at most $r^2$ stabilizer projectors; for a nonmember, a rational Hermitian witness valid on every stabilizer state.
- A one-step compression obstruction at the quadratic dimension scale and an extension to odd-prime local dimensions.

Here $r$ is the ordinary matrix rank, not the stabilizer rank. The complexity bound is established in the manuscript using polynomial-time rational linear programming; the Python implementation uses an exact rational simplex method and is intended for verification and experimentation.

## Run the qubit implementation

Use a Python environment with Python 3.10 and the versions listed in `requirements.txt`. From this repository root, run

```sh
python -m pip install -r requirements.txt
python verify_pauli_compression.py
```

The checks cover 1,800 two-qubit Pauli postselections, exact comparisons of support enumerations, a rank-eight example, a positive decomposition and a separating witness tested on every two-qubit stabilizer ray.

```python
import sympy as sp
from pauli_compression import stabilizer_membership
from stabilizer_lp import global_magic_witness, materialize_witness

eye = sp.eye(4)
v = eye[:, 1] + 2 * eye[:, 2]
rho = (eye[:, 0] * eye[:, 0].H + v * v.H) / 6
support, decision = stabilizer_membership(rho)
rows = support["pivot_rows"]
witness = materialize_witness(global_magic_witness(support, rho[rows, rows], decision))
assert sp.trace(witness * rho) == -sp.Rational(1, 6)
```

Pass exact SymPy integers and rationals, optionally with `sp.I`. The returned positive coefficients multiply unnormalized phase-vector outer products. To obtain probabilities, multiply each coefficient by the squared norm of its vector. `stabilizer_lp.py` uses a rational phase-I simplex with Bland's rule and verifies the certificates exactly. Its global witness uses a scaled orthogonal support projector. The paper gives an integer-matrix penalty as a second construction and proves the bit bound using polynomial-time rational linear programming.

## Build the local Lean project

With elan installed, run the following commands in this repository. The project pins Lean 4.24.0 and mathlib in its manifest.

```sh
cd lean
lake exe cache get
lake build
```

`QuantumOpen/PauliCompression.lean` proves Weyl Parseval, the average compressed energy, and injectivity of the two qubit Pauli branches. The enumeration, LP algorithm, complexity bound and odd-prime extension are proved in the manuscript. The Python implementation targets qubits.

## Typeset the paper

The repository includes the PDF and its LaTeX source. From the repository root, run

```sh
cd paper
tectonic main.tex
```

The figure source is `figures/draw_support.py`. The vector figure is included with the LaTeX source. To redraw it, use Matplotlib (tested with 3.9.2) and run `python figures/draw_support.py` from the repository root. Figure preparation follows the Matplotlib guidance in Kassis, Agarwal, He, Patel and Brueckner (2026). [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)

## Citation and contact

**Author:** SMC  
**Contact:** [changSM@buaa.edu.cn](mailto:changSM@buaa.edu.cn)

Please cite the manuscript if you use its results, algorithms, proofs, or implementation in your research. Feedback, corrections and related work are welcome via GitHub Issues or email. If you find the repository useful, please consider giving it a star.

SMC. (2026). *Exact Stabilizer Membership for Low-Rank Density Matrices*. Manuscript.

```bibtex
@unpublished{SMC2026StabilizerMembership,
  author = {SMC},
  title = {Exact Stabilizer Membership for Low-Rank Density Matrices},
  year = {2026},
  note = {Manuscript},
  url = {https://github.com/chang233333/low-rank-stabilizer-membership}
}
```

## Keywords

Stabilizer polytope, stabilizer membership, quantum magic, stabilizer states, low-rank density matrices, Pauli postselection, exact algorithms, magic witnesses, computational complexity, quantum resource theory, rational linear programming, Lean 4.
