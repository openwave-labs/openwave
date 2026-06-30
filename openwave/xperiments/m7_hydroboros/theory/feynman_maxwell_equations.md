# Feynman Lectures II, Ch 18 , The Maxwell Equations (the Maxwell baseline for M7)

> **Provenance.** Marc pointed Rodrigo here (2026-06-30): R. Feynman, *The Feynman Lectures on
> Physics*, Vol. II, **Chapter 18 "The Maxwell Equations"** (freely readable at
> feynmanlectures.caltech.edu/II_18.html). Feynman presents Maxwell's equations as the crowning
> achievement of 19th-century physics and shows how their interconnection reveals the speed of light.
> This note captures the key content as the **Maxwell baseline M7 must reproduce** (the EM-waves
> MODELS.md cell, M7.2). It is a textbook reference, not new physics.

## 1. The complete set (Feynman's Table 18-1)

Feynman writes the four equations in `ε₀ / c²` form (with `c² = 1/(ε₀μ₀)`):

```text
I.    ∇·E  = ρ/ε₀                  Gauss's law (charge is the source of E)
II.   ∇×E  = −∂B/∂t                Faraday's law (a changing B makes E)
III.  ∇·B  = 0                     no magnetic monopoles
IV.   c²∇×B = ∂E/∂t + j/ε₀         Ampère's law + Maxwell's new ∂E/∂t term
```

plus the **conservation of charge** (continuity), which is not independent but a consequence:

```text
∇·j = −∂ρ/∂t
```

## 2. Why Maxwell's new term is forced (the `∂E/∂t` displacement current)

The pre-Maxwell Ampère law was `c²∇×B = j/ε₀`. Take its divergence: the left side is `∇·(∇×B) = 0`
identically (divergence of a curl), so it would force `∇·j = 0`. But charge conservation requires
`∇·j = −∂ρ/∂t ≠ 0` whenever the charge density changes. **The equations were inconsistent.** Maxwell's
addition of `∂E/∂t` repairs it: take the divergence of IV,

```text
0 = ∂(∇·E)/∂t + (∇·j)/ε₀ = ∂(ρ/ε₀)/∂t + (∇·j)/ε₀   ⇒   ∂ρ/∂t + ∇·j = 0
```

so continuity is recovered automatically. The displacement current is **required by charge
conservation**, not an optional refinement.

## 3. The interconnection reveals `c` (light is an EM wave)

In free space (`ρ = 0, j = 0`), II and IV interlock: a changing `B` makes `E` (II), a changing `E`
makes `B` (IV). Combining the two first-order equations gives a **second-order wave equation**

```text
∇²E = (1/c²) ∂²E/∂t²        (and the same for B)
```

with propagation speed `c = 1/√(ε₀μ₀)`. The constant that entered as a static electric/magnetic units
ratio comes back out as the **speed of light**, so the very structure of the four equations predicts
that light is an electromagnetic wave. Feynman's § 18-4 "A traveling field" makes this concrete: an
infinite sheet of current switched on produces a slab of `E` and `B` that travels outward at exactly
`c`, a field that propagates with no charges or currents out ahead of it.

## 4. "All of classical physics" (§ 18-3)

Feynman's point: the whole of classical physics fits in one small table , Maxwell + charge
conservation + the Lorentz force + the relativistic law of motion + gravitation:

```text
Maxwell:        I-IV above
charge cons.:   ∇·j = −∂ρ/∂t
Lorentz force:  F = q(E + v×B)
law of motion:  d/dt [ m v / √(1 − v²/c²) ] = F
gravitation:    F = −G m m' / r²  ê_r
```

## 5. Relevance to M7 / HydroBoros

| M7 piece | What Feynman Ch 18 anchors | Plan ref |
| --- | --- | --- |
| **EM waves (Maxwell) cell** | the four equations are exactly what M7.2 must reproduce on the lattice (the MODELS.md "EM waves (Maxwell)" criterion) | [`../research/0a_implementation_plan.md`](../research/0a_implementation_plan.md) § 6 M7.2, § 7 |
| **Fleury's charge `= ∇·E`** | Gauss's law I: charge is literally the divergence of `E`. Fleury's "geometric charge" and the variable-λ Beltrami divergence are this same `∇·E` | § 0 (Trkalian/variable-λ), § 1 |
| **The self-sustaining toroidal circulation** | the II ↔ IV interlock (`E` feeds `B` feeds `E`) is the engine of the rotating/Beltrami vortex , the "snake eating its tail" at the field level | § 0 thesis |
| **`c` is not put in by hand** | M7's substrate is fundamentally Maxwell, so `c` and the wave structure must emerge, not be imposed; the Faraday constraint `ω = 2c/R₀` (Fleury) is the toroidal eigenvalue of exactly this system | § 1 (Faraday row), § 5 |
| **A-primary check** | Feynman's `E, B` are the fields; under M7's A-primary ontology these are `F = dA`, and I-IV are the consistency the potential `A` must satisfy | § 4 (A-primary) |

## Cross-references

- Implementation plan: [`../research/0a_implementation_plan.md`](../research/0a_implementation_plan.md)
- Companion theory: [arXiv:2510.22384](https://arxiv.org/abs/2510.22384) (Fleury's toroidal Maxwell solution),
  [`ceperley_rotating_waves.md`](ceperley_rotating_waves.md) (rotating-wave EM),
  [`sato_yamada_beltrami.md`](sato_yamada_beltrami.md) (Beltrami / force-free fields)
