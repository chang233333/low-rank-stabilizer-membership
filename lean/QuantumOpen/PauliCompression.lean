import Mathlib.Analysis.Fourier.FiniteAbelian.PontryaginDuality
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

open scoped BigOperators ComplexConjugate InnerProductSpace

namespace QuantumOpen

noncomputable section

variable {A : Type*} [AddCommGroup A] [Fintype A]

private theorem character_conj (χ : AddChar A ℂ) (x : A) :
    conj (χ x) = χ (-x) := by
  have h₁ : χ x * χ (-x) = 1 := by
    rw [← AddChar.map_add_eq_mul, add_neg_cancel, AddChar.map_zero_eq_one]
  have h₂ : χ x * conj (χ x) = 1 := by
    rw [Complex.mul_conj, Complex.normSq_eq_norm_sq, χ.norm_apply]
    norm_num
  have hn : χ x ≠ 0 := by
    intro h
    simp [h] at h₁
  exact mul_left_cancel₀ hn (h₂.trans h₁.symm)

private theorem character_parseval (f : A → ℂ) :
    ∑ χ : AddChar A ℂ, ‖∑ x : A, χ x * f x‖ ^ 2 =
      (Fintype.card A : ℝ) * ∑ x : A, ‖f x‖ ^ 2 := by
  classical
  have h : ∑ χ : AddChar A ℂ, (∑ x : A, χ x * f x) * conj (∑ y : A, χ y * f y) =
      (Fintype.card A : ℂ) * ∑ x : A, f x * conj (f x) := by
    calc
      _ = ∑ χ : AddChar A ℂ, ∑ x : A, ∑ y : A,
          (χ x * f x) * (conj (χ y) * conj (f y)) := by
        refine Finset.sum_congr rfl fun χ _ => ?_
        rw [map_sum, Finset.sum_mul]
        refine Finset.sum_congr rfl fun x _ => ?_
        simp only [map_mul, Finset.mul_sum]
      _ = ∑ x : A, ∑ y : A, ∑ χ : AddChar A ℂ,
          (χ x * f x) * (conj (χ y) * conj (f y)) := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun _ _ => ?_
        exact Finset.sum_comm
      _ = _ := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun x _ => ?_
        calc
          _ = ∑ y : A, (∑ χ : AddChar A ℂ, χ (x - y)) * (f x * conj (f y)) := by
            refine Finset.sum_congr rfl fun y _ => ?_
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun χ _ => ?_
            rw [sub_eq_add_neg, AddChar.map_add_eq_mul, ← character_conj]
            ring
          _ = _ := by simp [AddChar.sum_apply_eq_ite, sub_eq_zero]
  apply Complex.ofReal_injective
  simpa only [Complex.mul_conj, Complex.normSq_eq_norm_sq, ← Complex.ofReal_sum,
    ← Complex.ofReal_natCast, ← Complex.ofReal_mul] using h

/-- The translation and modulation operator indexed by a shift and a character. -/
def weyl (a : A) (χ : AddChar A ℂ) :
    EuclideanSpace ℂ A →ₗ[ℂ] EuclideanSpace ℂ A where
  toFun v := WithLp.toLp 2 (fun x => χ x * v (x + a))
  map_add' v w := by
    ext x : 1
    change χ x * (v (x + a) + w (x + a)) =
      χ x * v (x + a) + χ x * w (x + a)
    exact mul_add _ _ _
  map_smul' c v := by
    ext x : 1
    change χ x * (c * v (x + a)) = c * (χ x * v (x + a))
    ring

@[simp] theorem weyl_apply (a : A) (χ : AddChar A ℂ)
    (v : EuclideanSpace ℂ A) (x : A) :
    weyl a χ v x = χ x * v (x + a) := rfl

/-- The Weyl matrix coefficients form a tight frame with frame constant `card A`. -/
theorem weyl_parseval (v w : EuclideanSpace ℂ A) :
    ∑ p : A × AddChar A ℂ, ‖⟪v, weyl p.1 p.2 w⟫_ℂ‖ ^ 2 =
      (Fintype.card A : ℝ) * ‖v‖ ^ 2 * ‖w‖ ^ 2 := by
  classical
  have hinner (a : A) (χ : AddChar A ℂ) :
      ⟪v, weyl a χ w⟫_ℂ = ∑ x : A, χ x * (conj (v x) * w (x + a)) := by
    rw [PiLp.inner_apply]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [RCLike.inner_apply, weyl_apply]
    ring
  have hshift (x : A) : ∑ a : A, ‖w (x + a)‖ ^ 2 = ∑ a : A, ‖w a‖ ^ 2 :=
    Equiv.sum_comp (Equiv.addLeft x) (fun a : A => ‖w a‖ ^ 2)
  rw [Fintype.sum_prod_type]
  simp_rw [hinner, character_parseval, norm_mul, Complex.norm_conj, mul_pow]
  rw [← Finset.mul_sum, Finset.sum_comm]
  simp_rw [← Finset.mul_sum, hshift]
  rw [← Finset.sum_mul, ← EuclideanSpace.norm_sq_eq, ← EuclideanSpace.norm_sq_eq, mul_assoc]

variable {ι : Type*} [Fintype ι]

/-- Parseval for a finite family of coefficients, such as entries of compressed
operators in an orthonormal basis. -/
theorem character_parseval_family (f : A → ι → ℂ) :
    ∑ χ : AddChar A ℂ, ∑ i, ‖∑ x : A, χ x * f x i‖ ^ 2 =
      (Fintype.card A : ℝ) * ∑ x : A, ∑ i, ‖f x i‖ ^ 2 := by
  classical
  rw [Finset.sum_comm]
  simp_rw [character_parseval]
  rw [← Finset.mul_sum, Finset.sum_comm]

/-- The squared Hilbert–Schmidt norm of the compression to an orthonormal family. -/
def compressionEnergy (u : ι → EuclideanSpace ℂ A) (p : A × AddChar A ℂ) : ℝ :=
  ∑ ij : ι × ι, ‖⟪u ij.1, weyl p.1 p.2 (u ij.2)⟫_ℂ‖ ^ 2

theorem sum_compressionEnergy {u : ι → EuclideanSpace ℂ A} (hu : Orthonormal ℂ u) :
    ∑ p, compressionEnergy u p = (Fintype.card A : ℝ) * (Fintype.card ι : ℝ) ^ 2 := by
  classical
  unfold compressionEnergy
  rw [Finset.sum_comm]
  simp_rw [weyl_parseval, hu.norm_eq_one, one_pow, mul_one]
  simp [Fintype.card_prod, pow_two, mul_comm]

/-- A compression with energy less than one has no unit-modulus eigenvector in the span. -/
theorem coefficients_zero_of_eigenvalue {u : ι → EuclideanSpace ℂ A}
    (hu : Orthonormal ℂ u) (p : A × AddChar A ℂ) (henergy : compressionEnergy u p < 1)
    (c : ι → ℂ) (μ : ℂ) (hμ : ‖μ‖ = 1)
    (heigen : weyl p.1 p.2 (∑ j, c j • u j) = μ • ∑ j, c j • u j) : c = 0 := by
  classical
  let M : ι → ι → ℂ := fun i j => ⟪u i, weyl p.1 p.2 (u j)⟫_ℂ
  have hrow (i : ι) : ∑ j, M i j * c j = μ * c i := by
    have h := congrArg (fun v : EuclideanSpace ℂ A => ⟪u i, v⟫_ℂ) heigen
    simpa [M, map_sum, map_smul, inner_sum, inner_smul_right,
      hu.inner_right_fintype, mul_comm] using h
  let total : ℝ := ∑ j, ‖c j‖ ^ 2
  have hbound (i : ι) : ‖c i‖ ^ 2 ≤ (∑ j, ‖M i j‖ ^ 2) * total := by
    have htriangle : ‖∑ j, M i j * c j‖ ≤ ∑ j, ‖M i j‖ * ‖c j‖ := by
      simpa only [norm_mul] using norm_sum_le Finset.univ (fun j => M i j * c j)
    calc
      ‖c i‖ ^ 2 = ‖∑ j, M i j * c j‖ ^ 2 := by rw [hrow, norm_mul, hμ, one_mul]
      _ ≤ (∑ j, ‖M i j‖ * ‖c j‖) ^ 2 :=
        pow_le_pow_left₀ (norm_nonneg _) htriangle _
      _ ≤ _ := Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have htotal : total ≤ compressionEnergy u p * total := by
    simpa only [compressionEnergy, Fintype.sum_prod_type, Finset.sum_mul, M, total]
      using Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hbound i)
  by_contra hzero
  obtain ⟨i, hi⟩ : ∃ i, c i ≠ 0 := by
    by_contra! h
    exact hzero (funext h)
  have hpos : 0 < total := Finset.sum_pos'
    (fun j _ => sq_nonneg ‖c j‖) ⟨i, Finset.mem_univ _, by positivity⟩
  have hlt := mul_lt_mul_of_pos_right henergy hpos
  rw [one_mul] at hlt
  exact (not_lt_of_ge htotal) hlt

/-- An orthonormal family of size `r` in dimension greater than `r²` has a Weyl
operator with no unit-modulus eigenvectors in its span. -/
theorem exists_weyl_no_unit_eigenvalue {u : ι → EuclideanSpace ℂ A} (hu : Orthonormal ℂ u)
    (hdim : Fintype.card ι ^ 2 < Fintype.card A) :
    ∃ p : A × AddChar A ℂ, ∀ (c : ι → ℂ) (μ : ℂ), ‖μ‖ = 1 →
      weyl p.1 p.2 (∑ j, c j • u j) = μ • ∑ j, c j • u j → c = 0 := by
  classical
  have hex : ∃ p, compressionEnergy u p < 1 := by
    by_contra! h
    have hs : (Fintype.card A : ℝ) ^ 2 ≤ ∑ p, compressionEnergy u p := by
      simpa [Fintype.card_prod, pow_two] using
        (Finset.sum_le_sum (fun p (_ : p ∈ Finset.univ) => h p) :
          (∑ _p : A × AddChar A ℂ, (1 : ℝ)) ≤ ∑ p, compressionEnergy u p)
    rw [sum_compressionEnergy hu] at hs
    have hdim' : (Fintype.card ι : ℝ) ^ 2 < (Fintype.card A : ℝ) := by exact_mod_cast hdim
    have hpos : (0 : ℝ) < Fintype.card A := by positivity
    have hlt := mul_lt_mul_of_pos_left hdim' hpos
    exact (not_lt_of_ge hs) (by simpa [pow_two] using hlt)
  obtain ⟨p, hp⟩ := hex
  exact ⟨p, coefficients_zero_of_eigenvalue hu p hp⟩

/-- Every unit-phase branch of a suitable Weyl operator is injective on the span. -/
theorem exists_injective_weyl_branches {u : ι → EuclideanSpace ℂ A} (hu : Orthonormal ℂ u)
    (hdim : Fintype.card ι ^ 2 < Fintype.card A) :
    ∃ p : A × AddChar A ℂ, ∀ μ : ℂ, ‖μ‖ = 1 →
      Function.Injective (fun c : ι → ℂ =>
        weyl p.1 p.2 (∑ j, c j • u j) - μ • ∑ j, c j • u j) := by
  obtain ⟨p, hp⟩ := exists_weyl_no_unit_eigenvalue hu hdim
  refine ⟨p, fun μ hμ a b hab => ?_⟩
  change weyl p.1 p.2 (∑ j, a j • u j) - μ • ∑ j, a j • u j =
    weyl p.1 p.2 (∑ j, b j • u j) - μ • ∑ j, b j • u j at hab
  apply sub_eq_zero.mp
  apply hp (a - b) μ hμ
  simp only [Pi.sub_apply, sub_smul, Finset.sum_sub_distrib, map_sub, smul_sub]
  exact sub_eq_sub_iff_sub_eq_sub.mp hab

end

end QuantumOpen
