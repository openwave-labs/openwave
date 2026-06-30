# Sato & Yamada , Local Representation and Construction of Beltrami Fields

> **Provenance.** Shared by Marc Fleury (2026-06-30) as a HydroBoros Beltrami source.
> Source paper: N. Sato & M. Yamada, *Local Representation and Construction of Beltrami Fields*,
> [arXiv:1809.03136](https://arxiv.org/abs/1809.03136) (2018; Physica D, 2019). The PDF is kept local
> only (gitignored, not in the repo). The summary below is Marc's writeup (Gemini-assisted);
> the paper is the primary source, this note is orientation. Relevance to the M7 build is in § 5.

A Beltrami field `w` is an eigenvector of the curl operator with a (possibly space-varying)
proportionality factor `h`:

```text
∇ × w = h w
```

These are the **force-free** fields: steady Euler flows where velocity aligns with vorticity
(`v × ω = 0`), and force-free magnetic fields where `J × B = 0`. Sato & Yamada solved the long-standing
problem of how to **locally represent and systematically construct** a Beltrami field for a given,
potentially **inhomogeneous** `h`, by applying the **Lie-Darboux theorem** from differential geometry
to get a local Clebsch-like parametrization.

## 1. Core theoretical framework

A non-vanishing Beltrami field defines a differential 1-form that is **non-integrable** (a contact
form when `h ≠ 0`). Via the Lie-Darboux theorem, the field can locally be cast into a reduced,
standard canonical form. The key structural result:

> Locally, every smooth Beltrami field behaves like a reduced **Arnold-Beltrami-Childress (ABC)
> flow** with two of the standard parameters set to zero.

## 2. Local coordinate representation

Within a local neighborhood, a Beltrami field admits **two local invariants** (geometric coordinates):

1. a coordinate representing a physical plane of the flow, and
2. an angular-momentum-like quantity perpendicular to that plane.

This lets the field be written through a local **Clebsch-style parametrization**: instead of three
arbitrary spatial functions to track the field lines, the geometric constraints reduce the degrees of
freedom to specific scalar potentials satisfying a specialized metric configuration.

## 3. Construction method (the eikonal link)

The local representation becomes a concrete recipe: constructing a Beltrami field for a specified `h`
reduces to solving the **eikonal equation**

```text
|∇ψ|² = f(h, …)
```

### The equal-scale-factor rule (existence condition)

> The method guarantees the existence of smooth, nontrivial Beltrami fields for **any orthogonal
> coordinate system where at least two of the scale factors `hᵢ` are equal.**

So one can construct analytic, **solenoidal (`∇·w = 0`) and non-solenoidal** Beltrami fields in
standard geometries (cylindrical, spherical) even when `h` is **inhomogeneous** (spatially varying).

## 4. Physical applications

| Domain | What Beltrami representation buys |
| --- | --- |
| **Ideal MHD equilibria** | force-free fields `∇×B = μB` with spatially varying `μ`, matching real plasma configs better than constant-`λ` (Taylor-state) models |
| **Lagrangian turbulence** | with `v × ω = 0` the flow bypasses Bernoulli's constraint; field lines fill space ergodically, a structured handle on vortex reconnection + topological complexity |

## 5. Relevance to M7 / HydroBoros (why this matters for the build)

| M7 piece | How Sato-Yamada feeds it | Plan ref |
| --- | --- | --- |
| **The toroidal-Beltrami seeder** (M7.1) | a concrete construction recipe (eikonal + equal-scale-factor rule) for analytic Beltrami initial data on cylindrical / toroidal geometry, the seed for the self-linked vortex | [`../research/0a_implementation_plan.md`](../research/0a_implementation_plan.md) § 6 M7.1, § 5 |
| **The substrate question (Q1)** | the local Clebsch parametrization is exactly candidate **D** (the `(α,β)` / `ψ` seed-generator) made rigorous; informs whether Clebsch enters as a seeder vs an evolved DOF | § 9 Q1 |
| **Inhomogeneous `h` = non-monochromatic** | Fleury's torus fixes `ω = 2c/R₀` (constant `h = ω/c`); Sato-Yamada's **space-varying `h`** is the route past Fleury's single-frequency Heaviside ansatz toward a relaxed lattice field | § 1 (Fleury limit), § 6 M7.2 |
| **Non-solenoidal Beltrami (`∇·w ≠ 0`)** | their method covers **non-solenoidal** fields, the regime Fleury needs (charge `= ∇·E ≠ 0`); directly bears on Q5 (does a divergence-ful field still hold a clean knot?) | § 9 Q5 |
| **Force-free = the Ouroboros steady state** | `∇×w = hw` is the self-sustaining circulation the HydroBoros thesis identifies as the "snake eating its tail" | § 0 thesis |

## Cross-references

- Implementation plan: [`../research/0a_implementation_plan.md`](../research/0a_implementation_plan.md)
  (Task 0 source table § 3, Q1 / Q4 / Q5)
- Companion theory: [arXiv:2510.22384](https://arxiv.org/abs/2510.22384) (Fleury torus),
  Werbos *Evaluating Universe Model Alternatives v5* (shared doc, local only)
