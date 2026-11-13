# M10 background: what OPH has, what it lacks, and why the column exists

> **Purpose.** This is the gap map behind the M10 column: what Observer Patch
> Holography brings, what it is missing, and what a column for it is for. It was
> written by the maintainers at admission time (2026-11-13) from the author's
> application ([discussion #712](https://github.com/openwave-labs/openwave/discussions/712))
> and the author's repository at the pinned commit. The author owns the science;
> corrections by pull request or discussion are welcome.

## 1. What OPH is

OPH starts from observers, not from a field. An observer is a feedback loop: it
reads the ports of its patch, keeps a finite record, and its next action depends
on that record. The observers operate on a holographic screen of twelve-port
icosahedral carriers and run one local repair and memory law. Fields, particles
and spacetime are the effective descriptions the observers arrive at when they
compare records. The hypothesis behind it is closure: the structure reads itself
into existence, and time is the order observers find in their records.

That makes M10 unlike every other column. The emergence columns (M4 to M7) evolve
a field and look for particles in it. M8 reads a spectrum off a fixed topology.
M9 installs the Standard Model and works the gravity sector. M10 has no field at
the bottom at all: its native object is a set of update rules on a discrete
carrier, tier 2 in the terms of
[`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md). The specification of
record is [`m10_theory_canonical.md`](m10_theory_canonical.md).

## 2. The gap map (what the M10 program must supply)

| Gap | The statement | What supplies it |
| --- | --- | --- |
| No row runs on the carrier | By the author's own account, each criteria row runs on its own declared finite source (cubes, chains, two-cell blocks, several of them lattice field models by declaration) and none runs on the twelve-port screen | The join: one process carrying the rows. This is the column's defining open problem, and the author's hypothesis page sets the same condition ([`m10_theory_canonical.md`](m10_theory_canonical.md) § 5) |
| No genuine prediction | No number passes the prediction test of section 1.1, on the author's own audit: the lepton ratios, the hadron spectrum and the tau width ratio were compared with known data | A frozen number that is target-blind, distinct from the null and free of exposed choices. The author searched the executable model for one and found none |
| The lepton result is a fit | The register that gives Koide's 2/3 and the phase 2/9 was designed with the masses known, and its own controls show the alternatives (§ 4) | Frozen premises with a no-revision rule, then a comparison that could fail |
| The hadron source is calibrated | Seven measured inputs; the held-out comparisons fail | The failures are results and are on record. A derivation of the source from the carrier is open |
| Gravity has no physical scale | The one tested unit attachment misses G by 1.18 × 10⁴⁰ and gives light at 0.00196 c | Open, recorded by the author as a tested failure |
| No self-starting clock | The author's own theorem: an energy-conserving collision read with energy-compatible records cannot create the phase | Open, with the theorem on record |

## 3. Evidence weight (the grading the column starts with)

| Layer | Weight | Why |
| --- | --- | --- |
| The author's machine-checked theorems | author-side, not rebuilt here | Statements about declared finite objects (confluent repair, event order, a rank-three band, fluctuation identities). The Lean build and its theorem count were not reproduced at admission |
| The row constructions | author-side, inherited | Finite constructions for every row, on declared or calibrated sources. None was run in-platform, so every cell opens at 🚧 |
| The lepton headline numbers | recomputed at admission, from the formulas | The identity and the ratios reproduce (§ 4). That confirms the arithmetic, not the premises |
| The frozen prediction rows | registered falsifiers, weak by the author's own account | Each rejects a branch; none is target-blind and distinct from the null at once |
| The papers | preprint-tier | No paper is refereed (the author's statement); recorded in [`../theory/_CITATIONS.md`](../theory/_CITATIONS.md), as for M8 and M9 |

The practical rule for M10 tasks: **one row at a time, on the carrier's own
process, with the supplied pieces named.** The column's credibility is the join,
not the count of rows addressed.

## 4. Onboarding evaluation of record (2026-11-13)

| Check | Result |
| --- | --- |
| Provenance | Author account and repository real and active; the application is a complete STEP 2 briefing with the STEP 1 self-evaluation attached. All 85 repository links in the application resolve: 83 at commit `3c4e1e2`, the hypothesis page at `da76715` (byte-identical at both commits), and the root link. The Zenodo DOI resolves to a record with the stated title. These were checked by the maintainers at scaffold time; the papers themselves were not read |
| [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md) STEP 1 fit | Maps onto the criteria only through per-row declared sources. "Predictions exist" fails on the author's own audit. Reproducible in form: every number has a script and a receipt in the author's repository. Falsifiable at branch level: frozen rows with kill rules at named experiments. The five standard objections are answered |
| Tier | Tier 2: a local repair and memory law written as update rules is a closed system that runs. Tier 4, no closed system, is the only disqualifying case |
| Claim-to-artifact | The headline lepton numbers were recomputed by the maintainers from the stated formulas, with no author code run, and agree with the application (table below). A second, independent implementation agreed on every number |
| Parameter count | Run by the author in advance, and it finds a fit: at least five choices against two observables. The maintainers' count adds where the freedom sits (below) |
| Red flags | Self-reported by the author: knobs dressed as structure, reproduction of known coincidences (Koide, the Georgi-Glashow generation, the quark-model 5/3), unrefereed sourcing, declarative voice on the public site and in the book |
| Not checked | The author's code and tests; the Lean build and its theorem count; the papers; the packages behind the other rows; the timestamp custody of the frozen rows. The frozen dispersion values were deliberately left unread, so that [M10.1](tasks/m10_1_task_details.md) runs blind |

**The recompute.** Reference values: CODATA 2022 and the PDG tau mass. With
`sqrt(m_k)` proportional to `1 + sqrt2 cos(θ + 2πk/3)`:

| Quantity | The application | Recomputed |
| --- | --- | --- |
| Koide quotient of a circulant | 1/3 + (2/3)(\|b\|/a)² | same, symbolic; 2/3 at \|b\|/a = 1/√2 |
| m_μ/m_e at phase 2/9 | 206.7703 | 206.770316 (+9.83 ppm, 442σ) |
| m_τ/m_e at phase 2/9 | 3477.47 | 3477.4728 (tau at 1776.985 MeV, +0.61σ) |
| Tau from measured m_e and m_μ, quotient 2/3 | 1776.969027 MeV | 1776.969027 MeV (+0.43σ) |
| n_s = 1 − P/48, P = φ + √π α | 0.966021495637 | 0.966021495637 |

**Where the freedom sits in the lepton register.** The phase 2/9 rests on
premises the register's own proof file lists: two independent registers, the
inverse tick, the exclusive read, kick coefficient one, and the limit of many
rounds per clock unit. The file reports what three of the alternatives give, and
the mass ratios follow from those phases:

| Variant | What the proof file reports | m_μ/m_e (recomputed) |
| --- | --- | --- |
| Selected, many rounds | phase 2/9 | 206.770 |
| Ten rounds per clock unit | phase 0.222062 | 204.92 |
| One round per clock unit | phase 0.205334 | 92.29 |
| One shared register | count 1/4, not 2/9 | 2710 (reading the count as the phase) |

The ratio is steep in the phase: `d ln(m_μ/m_e)/dθ = 56`. At 302.28 rounds per
clock unit the muon ratio equals the measured one, so the round count acts as a
continuous knob, and the declared limit is what keeps the 442σ residual on the
record. The frozen tau row tests the balance and not the phase: the Koide value
1776.969 MeV and the register's 1776.985 MeV differ by 0.016 MeV, about a third
of the measurement precision that row asks for.

**Why admitted.** Admission asks whether the model hands the platform a closed
system that runs; cell status asks what has been derived. OPH's repair law on its
carrier runs, so the column opens. No row is derived on it, so every cell opens at
🚧. The K7 application ([discussion #525](https://github.com/openwave-labs/openwave/discussions/525))
stayed off the platform because it had neither a genuine prediction nor a closed
system to run; OPH comes through the second of the two doors that ruling named.

Admission terms of record (from the admission reply,
[discussioncomment-19595538](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19595538)):
the runs, scripts and AI tokens behind every claim are the author's; **one pull
request per concern**, from the author's fork with DCO sign-off, reviewed at
maintainer pace; **the author creates and orders the tasks on
[`m10_roadmap.md`](m10_roadmap.md)**, and every script, data file and finding
carries its task id; a result that will claim a cell files its gates and inputs
before the run; the run record stays in the author's repository at a pinned
commit, and the pull request brings the method note and a small script that
regenerates the headline number, with nothing imported in bulk; the column
scores one pinned candidate at a time; admission means a column with no cell
earned.
