# M5.52: the driven wave on the registered lines, the falsifier of the author's bands

> Staged 2026-11-23. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). It reads the falsifier that [M5.49](m5_49_task_details.md) registered as not read, on the same action, time part and backgrounds. Sister task of the same go: [M5.51](m5_51_task_details.md), the port of the author's newer action. The cell it reads for: the quantum wave equation (Klein-Gordon) in the M5 table of [`MODELS.md`](../../../../../MODELS.md); this plan changes no cell.

## TASK PLANNING (2026-11-23; staged)

**Scope.** The author registered the bands of the mid branch on their relaxed charge in their post of 2026-11-19 17:55 UTC, as edited at 18:25 UTC ([discussioncomment-19756045](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19756045)); the registration file is in the archive attached to their post of 2026-11-20 05:14 UTC ([discussioncomment-19766100](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19766100)). It names two falsifiers: "At ω = 4: a propagating (phase-advancing) wave along the axis", and "At either frequency: a turning radius off by more than 10 %". Under "Not predicted" it lists "The 3D wave pattern (local k near turning points, node positions, decay rates)". [M5.49](m5_49_task_details.md) read the bands from free noise, as local cutoffs on pieces of four lines, and did not read the first falsifier (its limit (3)). In their post of 2026-11-21 19:43 UTC ([discussioncomment-19800127](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19800127); the row of that date in the [thread record](m5_32_convo.md)) the author accepts the words of M5.49 and counts the entry "no band on the axis at ω = 4" as a yes or no statement registered within a few percent of its threshold: their least m_eff² on the axis is 16.58 against ω² = 16, and on OpenWave it is 15.50, 16.08 and 16.38 on half-boxes 8, 10 and 13 ([M5.49](m5_49_task_details.md) § 10). They name the driven-wave falsifier as the read they would value most, if such a run is cheap. This task drives the mid wave at ω = 4 and 5.5 on the action of [M5.42](m5_42_task_details.md) at J = 0 (κ = v = 1, δ = 0.05, L₁ = 0.03), with the time part of [M5.47](m5_47_task_details.md) § 1, on the three relaxed backgrounds that carry the labels of M5.49, and reads along the lines and pieces of M5.49:

| Reading | What is read | Against |
| --- | --- | --- |
| W1 | the falsifier: a propagating wave along the axis at ω = 4 | the registration (no band on the axis) |
| W2 | the same statement on the axis | F, the author's rule on the background of OpenWave |
| R2 to R4, N2 to N4 | the turning radii (the second falsifier), at ω = 4 in the plane and at ω = 5.5 on the axis and in the plane, as in M5.49 | F (R) and the registered numbers (N) |
| P1 to P3 | the phase carried across a band | F, with the offset of M5.49 as a stated interval: this plan's extension of F, not a test of the registration, which declares the local wavenumber not predicted |

**The backgrounds.** Grid b is the base state B0b of [M5.46](m5_46_task_details.md) (half-box 8, 65³), grid d is B0d of M5.46 (half-box 10, 81³), grid e is B0e of M5.49 (half-box 13, 105³), all at spacing 0.25 with the outer shell held, stored as `m5_46_B0b_end.npz`, `m5_46_B0d_end.npz` and `m5_49_B0e_end.npz` in [`../data/`](../data/) and loaded as the function `background` of [`m5_49_predict.py`](../scripts/m5_49_predict.py) loads them, the loop's center, normal and radius from the reader of [M5.44](m5_44_task_details.md). Grid a (B0a of M5.46, half-box 8, spacing 0.5, 33³) carries controls only.

**The author's driven run, and what is ported.** The archive of the post of 2026-11-20 05:14 UTC holds the script of the author's driven run of 2026-11-19 and its reader. They were read as text, the parts used here re-derived, nothing run.

| Element | The author's run (archive) | This plan |
| --- | --- | --- |
| Dynamics | linear response about the relaxed ring: the force $-[\nabla U(m_{bg} + \delta m) - \nabla U(m_{bg})]$, the kinetic matrix frozen at the background, a leapfrog step of 0.02 | the stepper of M5.47, unchanged (fourth-order Runge-Kutta, 𝒜 at the current field), with the base state's residual force subtracted (§ 1): the same linearization at first order, the background exactly at rest. Ported in substance |
| Background | their ring at L₁ = 0.03, half-box 12, spacing 0.5, the faces held at the deposited seed (not the exact vacuum), the absorbing layer within 2 of them | B0b, B0d, B0e at spacing 0.25, the shell held |
| Source | a Gaussian of width 1 at the origin, along their three-weight direction P, amplitude 10⁻³ h³, one frequency per run, a linear ramp over 15 time units | two circles of the loop's radius at ±3 along its normal, a Gaussian section of width 0.5, an acceleration along W, both frequencies in one run, a smooth ramp over 8 (§ 1). This plan's choice |
| Damping | $-\gamma(x) p$, with $\gamma = 3 \big( (2 - d)/2 \big)^2$ within 2 of the faces (d the distance to the nearest face), zero elsewhere | $-\gamma p$ with γ = 0.3 at every node, no layer (§ 1). The author's form, made uniform: this plan's choice |
| Read | the complex amplitude at ω over the last 30 time units on single nodes of +z, +x and the xz diagonal; the wavenumber as minus the radial gradient of its phase; the decay of its modulus | the lock-in amplitude over two windows of whole periods; the local complex wavenumber on the pieces of M5.49 and the local cutoff it gives (§ 3). The author's phase gradient is tabled as a descriptor only |

**Why the read is not a phase gradient, and why the changes.** Three facts of the registered geometry decide the design.

1. **Every band is closed.** The vacuum mass of the mid branch, √60.09 = 7.75, lies above both 4 and 5.5, and q falls toward 1 away from the loop, so between every band and the faces the mid wave is evanescent; so is the core of the loop (on the axis pieces nearest to its center m_F is 5.7 to 6.6 in the prediction of M5.49). A band is a resonator. In a steady driven state without loss its field is a standing pattern whose phase is uniform up to jumps of π at its nodes: the gradient of that phase is not a wavenumber. A layer at the faces does not change this: it lies beyond the evanescent shell and acts on the soft polarizations and on the tail. The author's own post-hoc fit in the archive, of an outgoing and a reflected wave on the axis at ω = 5, leaves a relative residual of 0.42 with or without the reflected part, so their read at that frequency does not tell the two apart either.
2. **A steady state needs loss.** Without it, the free modes that the ramp excites never decay, and a closed band at these frequencies holds many modes within the resolution of any window this task can afford.
3. **Near the threshold the literal phase cannot decide.** Under a damping strong enough to give a steady state, a wave whose ω² lies 0.5 above or 0.5 below its local cutoff has a complex wavenumber of one size, with a positive real part both ways (for c² = 0.87 and γ = 0.3 at ω = 4: 1.02 + 0.68 i and 0.68 + 1.02 i), so "the phase advances" holds on both sides. A front read before its first return fails the other way: telling ω² = 16 from 16.5 needs a frequency resolution of about 0.06 at ω = 4, hence a nearly monochromatic front some 100 time units long, during which a wave near its cutoff crosses a band of two units several times.

So this task reads the local dispersion, not the phase. A damping proportional to the momentum (the author's form, made uniform) gives a steady state in a fixed time and leaves the real part of the local law unchanged (§ 3); the estimator of M5.47 applied to the steady lock-in field reads that law at the driven frequency itself, whatever the pattern: standing, traveling or refracted. A source at the loop's center, where the author put theirs, would sit in the evanescent core: at ω = 4 it would reach the plane band only through more than two units of it, while the soft waves it also launches propagate freely. The source is moved inside the bands of both frequencies.

**Seven limits, stated before the run.** (1) The time part is that of M5.47 § 1, supplied by its plan; every word here is conditional on it, as in M5.49. (2) The action is that of M5.42, on which the author registered the bands, not the action of their later post, which the sister task [M5.51](m5_51_task_details.md) ports; no word carries over to that action. (3) One spacing, 0.25; no limit at zero spacing. (4) This is not a blind read. M5.49 read the local cutoffs on the same three backgrounds with another instrument, and its numbers set the expectations of § 6. What this read adds: the cutoff measured at ω itself, where M5.49's cutoff of a piece was an intercept extrapolated from the targets above it (its limit (3)); the falsifier in its own format, a driven wave; the phase across a band, and the decay beyond it as a descriptor. (5) The uniform damping, the subtracted residual force and the source are additions of this plan to the equations of M5.47 (§ 1). The read is of the real part of the local law, which the damping leaves unchanged when one local law holds over a piece. On a background that varies within a piece, terms with the gradient of the local speed enter Z, part of them through the flux that the damping sets; control C11 is its test, and the bias of a cutoff on a background that is not uniform stays not computed, as in M5.49. (6) Beyond the reach of the mid wave the read is not possible. Where the mid wave is evanescent and has decayed, the projection on W carries the soft waves that the kinetic form mixes in, which travel with no decay of their own; the ratio then returns the soft wave's mass, near 0, or a mixture of the two. The soft rule of § 3 removes a piece only where a soft wave dominates the projection; a piece whose projection holds a soft share of roughly 10 to 60 percent near the threshold can still read PROP with a wrong mass, and nothing in the read of a piece itself withholds it; the word WAVE of W1 alone carries a measured guard, the purity clause of § 6. What keeps the other words where the mid wave dominates is the near-threshold domain of § 3, set from F before the run: a prediction of where the mid wave reaches, not a measurement. The soft rule costs this: a true mid cutoff under about 0.62 of F would come out not read, never PROP. (7) The wording of the falsifier gives no measure of "propagating" or of how much phase makes a wave; the rule of W1 (§ 6) is this plan's.

**Definition of done.** The prediction file written, and its hashes noted in the run log, before the first driven job starts (C5); the controls of § 5 each met, or reported with the exact failing check; the jobs of § 2 ended at their length or reported as capped; every entry of § 6 carrying one word, or "no label" with the rule that withheld it; the expectations of § 6 scored against the outcome; the findings embedding, per line and labeled grid, the profile of y_read with its budget beside F, the interval of the offset and the cutoffs that M5.49 read, and the axis at ω = 4 on the three grids side by side; a record audit in a fresh context and a numbers check on a different model, each with its own code on the arrays, having tried to refute each word; the doc checker and the roadmap linter clean.

**Gating.** The user's go, one go with M5.51. M5.49 closed on 2026-11-20. No answer of the author is waited for. The plan audit and its fold are recorded at the end of this section.

**Research body.** This document (plan, findings, review); in [`../scripts/`](../scripts/): `m5_52_drive.py` (the prediction, the driver and the jobs; it imports `m5_47_evolve.py`, `m5_47_waves.py` and `m5_49_predict.py` and edits none of them) and `m5_52_reads.py` (the controls, the reads, the words, the tables and the plots); in [`../data/`](../data/): the prediction `m5_52_drive_predicted.json` and the summary `m5_52_summary.json` (tracked, each under 1 MB); plots `m5_52_*.png` in [`../plots/`](../plots/); the lock-in amplitudes, end states, energy series, checkpoints and the run log in `../checkpoints/m5_52/` (local-only, gitignored).

### 1. The driven equations

The stepper of [`m5_47_evolve.py`](../scripts/m5_47_evolve.py) is imported and not edited. Its `Field.rhs` returns, at every node off the shell, the rate $R = h^{-3} \mathcal{A}(M)^{-1} p$ and the force $f(M, p) = \partial T / \partial M - \partial U / \partial M$ of the action of M5.42 with the time part of M5.47 § 1, both zero on the shell. The driver of `m5_52_drive.py` uses them inside each of the four stages of the classical Runge-Kutta step of `RK4.step`, written again in the driver with the added terms and the same order of accumulation:

$$
\dot M = R , \qquad \dot p = f(M, p) - f_0 + s(x, t) - \gamma p , \qquad f_0 = f(M_0, 0)
$$

on the nodes off the shell (zero on the shell, as in M5.47), from rest: M = M₀ and p = 0 at t = 0.

| Term | Definition | Why |
| --- | --- | --- |
| $f_0$ | the force of the base state at rest, held constant (the states are at the gate of M5.45, not at rest) | M₀ becomes an exact rest state of the driven equations, so no unperturbed twin is run; the author's linearization does the same. Controls C3 and C4 |
| $s(x, t)$ | the source, below | the drive |
| $-\gamma p$ | γ = 0.3 at every node off the shell (0.15 in job GbG) | the damping is proportional to the inertia, so every free mode above γ/2 decays as $e^{-\gamma t / 2}$; overdamped soft box modes decay more slowly and lie eight Hann bins from the tones. The real part of the local law is unchanged (§ 3) |

**The source.**

$$
s(x, t) = a_s r(t) \big( \sin 4t + \sin 5.5t \big) g(x) h^3 \mathcal{A}(M_0(x)) \hat W(x)
$$

with Ŵ(x) the six components of the mid polarization W of M5.47 § 3 (the direction along which M5.47 started its unit jobs) and 𝒜(M₀) the kinetic matrix of the base state at each node (`Field.a_matrices`), so that the source alone accelerates the field along W. The amplitude is a_s = 10⁻⁴ (5 × 10⁻⁵ in job GaH); the ramp is $r(t) = \sin^2(\pi t / 16)$ for t under 8 and 1 after. The profile is two circles:

$$
g(x) = \sum_{\pm} \exp \Big( - \frac{(\rho - R_s)^2 + (z \mp z_s)^2}{2 \sigma_s^2} \Big)
$$

with ρ and z the cylindrical coordinates about the loop's center and normal, R_s the loop's radius (3.3 in the vacuum jobs, about the center of the box with the normal along the third axis), z_s = 3.0, σ_s = 0.5, and g set to zero on the shell. Checked at plan time on the three base states (q only, no wave): q is 4.0 to 4.2 on the two circles, above the threshold of ω = 4 (3.7556) and of ω = 5.5 (1.9864), so the source sits inside both bands where W is defined (the two largest eigenvalues 0.87 apart), and by the geometry no node of a piece of lines A, PF or PD has g at 10⁻³ or more, on b, d, e or in the vacuum; line O loses its pieces from about 2.25 to 6.25. The width 0.5 weights a wavenumber k of the source by $e^{-k^2 \sigma_s^2 / 2}$: 0.14 at k = 4 and 0.02 at k = 5.5, where the soft waves of the drive lie, against 0.75 at k = 1.5.

**The energy.** With T = ½ Σ R · p as coded (sums over the nodes and the six components):

$$
H' = T + U(M) + f_0 \cdot (M - M_0) , \qquad H'(t) - H'(0) = \int_0^t \Big( \sum s \cdot R - 2 \gamma T \Big) dt'
$$

The work of the source $\mathcal{W}(t)$ and the loss $\mathcal{L}(t)$, the two terms of the integral, are accumulated with the weights (1, 2, 2, 1)/6 of the four stages of each step.

**The steady state at a frequency, to first order** (used by C12). With $M - M_0 = \operatorname{Re} \big( A e^{-i \omega t} \big)$ at the drive ω, H the Hessian of U at M₀ (∂T/∂M is of second order about rest) and $S = a_s g h^3 \mathcal{A}(M_0) \hat W$:

$$
\big( \omega^2 + i \omega \gamma \big) h^3 \mathcal{A}(M_0) A = H A - i S
$$

**The step and the windows.** The step is $\Delta t = 4\pi / 628 \approx 0.02001$, the step of record of M5.47 on the unit adjusted so that 4π, a common period of the two tones (8 periods of ω = 4, 11 of ω = 5.5), holds 628 steps. A job runs 3356 steps: the ramp, a settling to step 2100 (t ≈ 42.0), then two windows of 628 steps. The lock-in amplitude of window k at a tone is

$$
A_k(x, \omega) = \frac{2}{\sum_j w_j} \sum_{j = j_k}^{j_k + 627} w_j e^{i \omega t_j} \big( M(x, t_j) - M_0(x) \big) , \qquad w_j = \sin^2 \frac{\pi (j - j_k)}{628}
$$

with j₁ = 2100 and j₂ = 2728, so that $M - M_0 = \operatorname{Re}(A e^{-i \omega t})$ for a steady tone. Over whole periods of both tones the periodic Hann window returns a steady tone exactly and makes the two tones exactly orthogonal. From the end of the ramp, what remains of a free mode above γ/2 is $e^{-5.1}$ of its size at the start of window 1 and $e^{-7.0}$ at the start of window 2, so a transient that matters shows as a difference of the two windows (C10).

### 2. The jobs

| Job | Background | Grid | Settings | Role |
| --- | --- | --- | --- | --- |
| Ge | B0e | half-box 13, 105³ | the settings of § 1 (γ = 0.3, a_s = 10⁻⁴, 3356 steps) | the words of grid e; the only grid that holds the outer edges at ω = 5.5 |
| Gd | B0d | half-box 10, 81³ | the same | the words of grid d |
| Gb | B0b | half-box 8, 65³ | the same | the words of grid b |
| GbG | B0b | half-box 8, 65³ | γ = 0.15; settling to step 4200, windows from steps 4200 and 4828, 5456 steps | C11, the damping |
| VD1, VD2, VD3, VD4 | the uniform vacuum of M5.47 (`m5_47_waves.vacuum_matrix`) | half-box 6, 49³ | v = 0.553761, 0.250295, 0.282248, 0.473214, so that the mid mass squared is 33.275 (1.1 × 30.25), 15.04 (0.94 × 16), 16.96 (1.06 × 16) and 28.435 (0.94 × 30.25); otherwise § 1 | C9, the known answer (§ 5) |
| Ga0 | B0a | half-box 8, spacing 0.5, 33³ | § 1 | the first driven job (stage 0); the reference of C4, C6, C7 |
| GaH | B0a | 33³ | a_s = 5 × 10⁻⁵ | C6, linearity |
| GaS | B0a | 33³ | step 4π/1256, every step count doubled (the same times) | C7, the step |
| GaP, GaN | B0a | 33³ | f₀ not subtracted; GaN with a_s = 0 (its unperturbed twin) | C4, the subtraction against a twin |

On the vacuum jobs the heavy branch, up to 262 in mass squared, keeps ω Δt under 0.5. The lines of a vacuum job run from the center of the box with the normal along the third axis, as the vacuum blocks of M5.49.

**Per job.** E, T, U, $H'$, $\mathcal{W}$ and $\mathcal{L}$ every 20 steps; the largest $\lvert M - M_0 \rvert$; the lock-in amplitudes of both windows at both tones (six components, complex) and the end state, in `../checkpoints/m5_52/`; a restartable checkpoint every 200 steps on grids of 65³ and above. Wall caps, as run rules: Ge 7 hours, Gd and GbG 4 hours, Gb 3 hours, each vacuum job 2 hours, each job of grid a 1 hour. A job that reaches its cap checkpoints, stops and is reported as capped.

**The scope cut, decided now.** If Ge is capped, the entries of grid e carry no label ("capped"): R3, R4, N3, N4, P2 and the P3 entries of grid e carry none (only grid e holds them, but for P3 on PD of d), and W1 and W2 rest on grids b and d.

### 3. The read

**Polarization, lines and pieces: those of M5.49 § 2, unchanged.** The scalar $a(x, \omega) = \langle W(x), A_2(x, \omega) \rangle$ with the inner product of M5.47 § 3; lines A, PF, PD and O from the loop's center; pieces of Δ = 0.5 with the tube t₀ = 1.0 and the cone of 15°; the domain D (1.5 inside every face, the two largest eigenvalues of the background at least 0.5 apart); a piece read when its nodes in D reach 150 and 90 percent of its nodes. They are computed with the functions of [`m5_49_predict.py`](../scripts/m5_49_predict.py) and checked equal to the geometry stored in [`m5_49_bands_predicted.json`](../data/m5_49_bands_predicted.json) (C14).

**The estimator** is that of M5.47 § 3 on a piece S_i, kept complex, on the unfiltered field:

$$
Z_i = - \frac{\sum_{x \in S_i} \bar a \Delta_h a}{\sum_{x \in S_i} \lvert a \rvert^2}
$$

with Δ_h the seven-point Laplacian. Its real part is the K² that M5.49 tabled at a target.

**The local law with damping.** For a wave of the mid branch that obeys the local law of M5.47 about a point, $\omega^2 = c^2 K^2 + m^2$ (m the local cutoff, c the local speed along the wave), the damping $-\gamma p$ turns ω² into ω² + iωγ:

$$
K^2 = \frac{\omega^2 - m^2 + i \omega \gamma}{c^2}
$$

The phase of K² does not depend on c. Where one local law holds over a piece, a pattern of any directions (standing, traveling or refracted) gives Z_i = K², and

$$
m^2_{\mathrm{read}} = \omega^2 - \omega \gamma \frac{\operatorname{Re} Z_i}{\operatorname{Im} Z_i} , \qquad c^2_{\mathrm{read}} = \frac{\omega \gamma}{\operatorname{Im} Z_i} , \qquad y_i = \omega^2 - m^2_{\mathrm{read}}
$$

so the local cutoff is read without the speed. The ratio is exact for one local law. Where the waves of a piece travel at two speeds (M5.47 § 3), the cross terms of Z_i carry an error that grows as y/(ωγ) deep in a band: for three plane waves per piece at c² of 0.7 or 0.9, the 90th percentile of the error is 0.14 at y = ±0.5 and ω = 4, 1.35 at y = 6, and 2.2 at y = 14 and ω = 5.5 (a made-up input, C13 (b)). The binned damping term of the budget carries it (below, C11). In the vacuum the estimator is exact on the lattice: Δ_h is the operator of the lattice law of M5.47 § 1 (C9). The convention of § 1 gives Im Z_i > 0; a reversed sign shows in C9 and C13.

**When a piece is NOT READ.** When any of its nodes has g ≥ 10⁻³ (the source); when the root mean square of a over its nodes in D is under 10⁻¹² (four orders above the round-off of a field of order one); when $c^2_{\mathrm{read}}$ lies outside 0.4 to 2.5 (the fitted slopes of the pieces of M5.49 on grids b, d and e lie from 0.79 to 1.62; the bound is the plan's choice, wide around them, and guards against a ratio with a vanishing imaginary part); or, the soft rule, when $m^2_{\mathrm{read}} \lt 0.5 (0.87 m_F)^2$ (in the vacuum, m_F² is the set mass squared). A soft wave alone returns a mass near 0, and the cutoffs that M5.49 read on its fitted pieces lie at 0.868 to 1.049 of m_F, all above the 0.62 of m_F at which the soft rule acts, so no piece that M5.49 read would be lost to it. A piece NOT READ (soft) counts as not read for every cover.

**The near-threshold domain.** At a frequency ω, the pieces of a line where F on the pieces has $m_F^2 \le 1.2 \omega^2$. It is the cover domain of W1 and bounds the stretch of P (§ 6). Its reason: a mid wave read at a depth where it has decayed is taken over by the soft waves (limit (6)), and the read cannot tell. On the axis at ω = 4 the domain is one contiguous stretch on every grid: s = 1.75 to 4.75 on b (7 pieces), 1.75 to 4.25 on d and e (6 pieces each). Across it, the largest decay of the mid wave that F predicts along the axis, from the piece of least m_F to either end, is 1.54 (b), 1.42 (d) and 1.68 (e) e-folds at λ = 1, and under 0.7 at λ = 0.935. With $m_F^2 \le 1.3 \omega^2$ the stretch would reach 2.61 e-folds (factor 13.6) on grid e, at which a soft wave of a few percent of the mid amplitude on W would equal it; that is why the bound is 1.2.

**The filter**, used only as a term of the budget. The binomial filter [1, 2, 1]/4 along each axis, applied 8 times per axis, after extending a past the shell by odd reflection about the shell nodes (the shell is held, so a = 0 there). It commutes with Δ_h, so a field that obeys $\Delta_h a = -K^2 a$ with one K² over the reach of the filter (two length units) keeps its Z; a wave along an axis is weighted by $\cos^{16}(k h / 2)$: 0.75 at k = 1.5, 0.12 at k = 4, 0.016 at k = 5.5. It also widens the source: on piece nodes the filtered g reaches about 9 × 10⁻³ where the raw g is at most 3.3 × 10⁻⁴. So the filtered value $y_i^{\mathrm{filt}}$ enters ε only on pieces where the filtered g is under 10⁻⁵ at every node; elsewhere that term is absent and the entry is marked so. By the geometry the term is absent on every piece of the vacuum jobs and of the axis domain of W1; it is present only on the outer pieces, from s of about 6.25 to 6.75 outward (on d: 6.25 to 8.25 on A, 6.75 to 8.25 on PF, 6.75 to 9.75 on PD), and on no piece of PF on b.

**The budget** of a piece, a frequency and a grid:

$$
\varepsilon_i = \max \Big( 0.01 \omega^2 , \quad 2 \big\lvert y_i - y_i^{(1)} \big\rvert , \quad 2 \big\lvert y_i - y_i^{\mathrm{filt}} \big\rvert , \quad 2 b_\gamma(\omega, x_i) , \quad 2 b_{\mathrm{vac}}(\omega) \Big)
$$

with y_i read on window 2, $y_i^{(1)}$ the same on window 1, $y_i^{\mathrm{filt}}$ window 2 filtered (where allowed, above), and $b_{\mathrm{vac}}(\omega)$ the worst $\lvert m^2_{\mathrm{read}} - m^2 \rvert$ over the read pieces of the vacuum cases of C9 that carry a bound at that ω. The damping term is binned by $x_i = \lvert y_i \rvert / (\omega \gamma)$: bin I under 1, bin II from 1 to 4, bin III 4 and above; $b_\gamma(\omega, x_i)$ is the 90th percentile of $\lvert y(\mathrm{Gb}) - y(\mathrm{GbG}) \rvert$ over the read pieces of grid b in the bin of the piece, and a bin with fewer than five pieces on grid b takes the largest value of all bins. The bin of each piece on every grid sets its term. The floor 0.01 ω² is the tolerance of C9 (the plan's choice). Every y is printed with its ε.

**Status and profiles.** On a line the three profiles are y, y + ε and y − ε, each joined by straight lines between the centers of consecutive read pieces no more than 2Δ apart (the rule of M5.49 § 2). On a profile p a read piece is PROP when p ≥ 0 and EVAN when p < 0: the margin is carried by the shift alone. A piece PROP on all three profiles (y − ε ≥ 0) is PROP; EVAN on all three (y + ε < 0) is EVAN; otherwise it is MARGIN. An edge is a radius where a profile crosses 0: outer where it goes from positive to negative as s grows, inner the other way. On a stretch 𝒮 of joined pieces:

$$
\Phi[p] = \int_{\mathcal{S}} \sqrt{\max \big( p(s), 0 \big)} ds , \qquad \bar\kappa[p] = \frac{1}{n} \sum_{i \in \text{window}} \sqrt{\max ( - p_i, 0 )}
$$

The same functionals are applied to F, with $y_F(\lambda) = \omega^2 - \lambda^2 m_F^2$ on the same pieces. The units are those of the author's law, whose speed is 1: the ratio removes the speed from the read, and the registration's rule has none.

**The regime of each read.**

| Read | Valid where | Not valid where, and the guard |
| --- | --- | --- |
| $m^2_{\mathrm{read}}$ and y on a piece | in D, away from the source, the field above its floor, in a steady state, one local law over the piece, the mid wave dominant on W | near the loop line, where W is not defined (D); near the source (the rule on g); beyond the reach of the mid wave (the near-threshold domain, the soft rule, limit (6)); deep in a band with two speeds (the binned damping term); where the cutoff changes within a piece (C10, C11) |
| Φ | on a stretch of read pieces inside the near-threshold domain, shared by F and the read | the literal phase of the field: under the damping its gradient is positive on both sides of a cutoff (Scope), so it is a descriptor |
| $\bar\kappa$ | a descriptor only, on read pieces within 1.5 of an outer edge of F | deeper tails, where the soft waves take over |
| An edge | between read pieces no more than 2Δ apart | under a radius of 2.5 (pieces of 0.5); next to the loop |
| The lock-in amplitude | after the settling, over whole windows | during the ramp; a window that still holds a transient (C10, C12) |

**Descriptors, with no word of their own.** Per piece: $c^2_{\mathrm{read}}$, Re Z_i (the K² of M5.49 at the target), the filtered and window 1 values, the root mean square of a, the share of the six-component power outside W, and the status of the pieces outside the near-threshold domain. The decay beyond an outer edge: $\bar\kappa$ of the read on the read pieces within 1.5 of the edge, tabled beside the interval of F from $\bar\kappa_F(0.87)$ to $\bar\kappa_F(1)$. Per ray of A, PF and PD: the author's read, the phase of a along the ray and minus its gradient, and the modulus of a. Made only after the words: Re Z of the stored free-noise sums of M5.49 at the targets 4 and 5.5 (Ub, Ud and Ue against their twins), on the same pieces, beside the driven Re Z; with no damping in those jobs the ratio does not apply to them.

### 4. The prediction, written first

`m5_52_drive.py --predict` writes `../data/m5_52_drive_predicted.json` before Ga0, the first driven job, starts. It opens the base states and the tracked prediction of M5.49, and nothing that a driven job writes. It checks the hashes of the blocks b, d, e and V2 of `m5_49_bands_predicted.json` against that file's `hashes` field, and stops on a mismatch.

| Entry | Definition |
| --- | --- |
| F | m_F per piece and on the ray (with q and with q_P), copied from the hashed blocks of M5.49, with C, the cutoffs of M5.47 per bin, beside it (no label) |
| y_F(λ) | $\omega^2 - \lambda^2 m_F^2$ per piece at ω = 4 and 5.5, for λ = 1, 0.935 and 0.87 |
| Edges of F | as in M5.49, on the pieces and on the ray |
| The near-threshold domain | per line and ω, the pieces with $m_F^2 \le 1.2 \omega^2$; on the axis at ω = 4 its pieces and its largest predicted decay |
| The statement of F on the axis at ω = 4 | by the rule of W1 (§ 6) on y_F(1), with ε = 0.01 ω², on the pieces and on the ray |
| Φ_F(λ) and $\bar\kappa_F(\lambda)$ | on the stretches and windows that the read pieces of M5.49 give, for reference; the words apply the same functionals to the stretches the driven read gives (§ 6) |
| The source | g and the filtered g on every piece, the excluded pieces, the pieces where the filter term is allowed, q on the two circles |
| The vacuum | the known answers: per VD job the mid mass, y and Φ on each line at each ω |
| The settings | every number of §§ 1 to 3 |

The SHA-256 of each block (keys sorted, separators `,` and `:`, UTF-8, floats with seven significant digits: the rule of M5.49) and of the whole file are written in the run log before Ga0 starts. `m5_52_reads.py` takes the block hashes as arguments and stops on a mismatch, and the predict mode has no code path that opens a lock-in amplitude (C5).

**The offset of M5.49 as an interval, decided now.** On the grids of spacing 0.25, the bins of M5.47 § 11 fitted cutoffs from 0.923 to 0.984 of $\sqrt{m^2 / \bar q}$. The fitted pieces of M5.49 with q̄ above 2 read 0.868 to 1.049 of F (on PF of grid b, for one, 0.868 to 0.906 next to the band of ω = 4); their medians lie 3.7 to 6.2 percent under F over all those pieces and 3.74 to 6.65 percent under over the pieces that lie under F; its HELD edges lie 3.4 to 7.4 percent outside the edges they were scored against. λ = 1 is the author's rule on the background of OpenWave; λ = 0.87, the least piece ratio the record holds, is the low end; λ = 0.935 is tabled as the middle value. The interval λ from 0.87 to 1 is what is known before the run, and it is not a correction: no word is computed at a single λ chosen after the read.

**At plan time**, from the tracked blocks b, d and e of M5.49, with code written for this plan (no wave was read): on the axis pieces at ω = 4 the largest y_F(1) is +0.50 (b), −0.08 (d) and −0.38 (e). Over the near-threshold domain of the axis, Φ_F is about 0.6, 0 and 0 radians at λ = 1; 3.35, 2.57 and 2.16 at λ = 0.935; 5.49, 4.44 and 4.26 at λ = 0.87; on the ray form over all the axis at λ = 1, 1.4, 0.6 and 0.2. The cutoffs that M5.49 read on those pieces, put in place of m_F, give 3.05, 1.82 and 1.16. With the floor of ε, NO WAVE needs the cutoff measured at ω = 4 to lie at or above 1.021, 1.003 and 0.993 of F over the whole domain (b, d, e), and WAVE needs it at or under about 0.935, 0.915 and 0.907 of F. The cutoffs that M5.49 read on the domain pieces of grid b lie at 0.92 to 1.00 of F. Elsewhere the phases are many radians: P2 on the axis of e at ω = 5.5 is about 20 at λ = 1 and 33 at λ = 0.87.

### 5. The controls

| Control | Input | The answer it must return | What a failure would look like |
| --- | --- | --- | --- |
| C1, the stepper | `m5_47_evolve.py --controls`, unchanged, its output written outside `../data/` | K0 to K5 of M5.47: the leaves of [`m5_47_controls_stepper.json`](../data/m5_47_controls_stepper.json) reproduced bit for bit (timing and series fields left out), as C1 of M5.49 | a stepper that is no longer the one M5.47 controlled |
| C2, the driver reduces to the stepper | the driver on B0a with a_s = 0, γ = 0 and f₀ not subtracted, 50 steps, against `RK4.step` of M5.47, with the order of accumulation of `RK4.step` | `numpy.array_equal` on M and p after every step | a wrapper that changes the step |
| C3, the hold (a wiring check) | the driver on B0a with a_s = 0, γ = 0.3 and f₀ subtracted, 500 steps | largest $\lvert M - M_0 \rvert$ at most 10⁻¹² | f₀ from another routine or another state than the force of the step. With f₀ from the same routine it is zero by construction and cannot fail for a physical reason |
| C4, the subtraction against a twin | GaP − GaN against Ga0: the lock-in amplitudes of window 2, both tones, six components, on the nodes of D of grid a with g under 10⁻³ | relative difference in norm at most 1 percent | a subtraction that changes the response, not only the background |
| C5, the prediction first | the prediction file, the run log, the reads script | block and file hashes in the run log before Ga0 starts; the reads take the hashes and stop on a mismatch; the hashes of the four blocks of M5.49 match | a prediction written or changed after a driven job |
| C6, linearity | Ga0 against GaH, both tones and both windows, on the same nodes | $\lVert A(\mathrm{Ga0}) - 2 A(\mathrm{GaH}) \rVert / \lVert A(\mathrm{Ga0}) \rVert$ at most 2 percent (K9 of M5.47) | a response not linear in the source, or two tones that mix |
| C7, the step | Ga0 against GaS, the same norm | at most 1 percent | a read set by the step |
| C8, the energy | every job: the series of § 1 | the largest $\lvert H'(t) - H'(0) - \mathcal{W}(t) + \mathcal{L}(t) \rvert$ at most 2 percent of the largest $\lvert H'(t) - H'(0) \rvert$ (the 2 percent of K11 of M5.47) | a source or a damping that the equations of § 1 do not contain: a sign, a term missing in one stage. A factor common to the source and to its work (h³, a_s, 𝒜) passes C8; C12 catches it, with S computed in the reads script from the formula of § 1, not taken from the driver |
| C9, the known answer | the vacuum jobs on the pieces of A, PF and PD (26 read pieces), read as § 3 reads the unit. At ω = 4: VD2 (propagating, y = +0.96) and VD3 (evanescent, y = −0.96). At ω = 5.5: VD1 (evanescent, y = −3.02) and VD4 (propagating, y = +1.82) near the threshold, and VD2 and VD3 (y = +15.21, +13.29) deep in the propagating range | on each of these six cases: at least 24 of the 26 pieces read, none removed by the soft rule; every read piece within 0.01 ω² of the set mass squared; $c^2_{\mathrm{read}}$ within 1 percent of 1; the status of every read piece as set, with ε = max(0.01 ω², 2 $b_{\mathrm{vac}}$) and no damping term (that term is measured on grid b and does not enter the vacuum control); no edge on any line; Φ within 2 percent of $\sqrt{y}$ times the stretch where propagating. Checked at plan time on an exact linear steady state of these jobs (§ 9): 26 of 26 read, none soft, the worst error 0.011 against tolerances of 0.16 and 0.30, c² 1.000 to 1.006 | a reader that returns one answer at a frequency 6 to 10 percent from its threshold; a reversed sign convention; a source term in the read pieces |
| C9, the deep evanescent cases | VD1 and VD4 at ω = 4 (y = −17.3 and −12.4), with no bound on the mass | tabled, not scored: every piece lies outside the near-threshold domain (m² of 2.08 and 1.78 ω²), so none enters a word | none: these cases show why the domain exists (§ 9) |
| C10, the windows | every job, every read piece, both tones | per grid and ω, the median of $\lvert y_i - y_i^{(1)} \rvert$ at most 0.01 ω² | a transient that sets the read |
| C11, the damping | Gb against GbG, the read pieces of grid b, both tones, per bin of § 3 | the 90th percentile of $\lvert y(\mathrm{Gb}) - y(\mathrm{GbG}) \rvert$ at most 0.01 ω² in bin I, 0.06 ω² in bin II and 0.12 ω² in bin III (the plan's choice, of the size that the two-speed input of C13 (b) gives; in bin III at ω = 4 that input alone comes near the bound) | a read that depends on the damping beyond what two speeds give: a term with the gradient of the speed of the size of the law |
| C12, the steady state | every job, tone and window, on the nodes of D with g under 10⁻³ | $\lVert H A - (\omega^2 + i \omega \gamma) h^3 \mathcal{A}(M_0) A - i S \rVert$ at most 3 percent of $\lVert H A \rVert$ (the 3 percent of K12 of M5.47), with HA from central differences of the coded static force about M₀ along Re A and Im A, scaled so that the largest change of a component is 10⁻⁵, and S computed in the reads script from the formula of § 1 | a lock-in amplitude that is not the steady response of the linearized equations: a transient, a term of higher order, a wrong convention, a common factor in the source |
| C13, the readers on made-up inputs | (a) lock-in: a pure two-tone series; a tone at 5.5 alone read at 4; a tone at 4.3 read at 4. (b) estimator, on a 49³ lattice of spacing 0.25, on nodes at least 2.5 from the faces: plane waves $e^{i K \cdot x}$ with the complex K that solves the lattice law with damping for m² = 14 and 18 at ω = 4 and γ = 0.3, along an axis and along a diagonal, and the sum of two opposite ones; the same filtered; the same with a real plane wave of $\lvert k \rvert$ = 4 carrying 10 percent of the power; a massless damped wave carrying 90 percent of the power over the evanescent wave of m² = 18; the two-speed input, read at γ and at γ/2: per draw of `numpy.random.default_rng(52013)`, three plane waves, each with a direction uniform on the sphere, a complex amplitude with standard normal real and imaginary parts, c² of 0.7 or 0.9 with equal chance, and the complex K of the lattice law with damping for m² = ω² − y; a piece is the nodes of a slab 0.5 thick inside a tube of radius 1; 300 draws at each y of −3.2, −1, −0.5, 0.5, 1, 3, 6 and 7 at ω = 4 and of −3.2, −1, 1, 3, 6, 10 and 14 at ω = 5.5. (c) made-up profiles of y with budgets | (a) the amplitudes to 10⁻¹²; under 10⁻¹² of the 5.5 tone; the analytic response of the window to 10⁻¹⁰. (b) $m^2_{\mathrm{read}}$ to 10⁻⁹ and the right status; filtered unchanged to 10⁻⁶; with the added wave, filtered within 0.01 ω² of 14 and of 18 and unfiltered off by more than 0.05 ω² (so the filter term can be large); the 90 percent soft input NOT READ (soft); on the two-speed input, in every bin at each ω, the error of $m^2_{\mathrm{read}}$ exceeds ε on at most one draw in ten, with ε built from the floor and the damping term of the same input (bin I through the floor); the 90th percentiles of the error are printed and are no pass criterion. (c) each outcome of § 6 once: W1 HELD, NOT HELD, and no label by a zone, by the margin, by the cover of the domain, by the three profiles, by window 1, by the purity clause; W2 each word; an edge HELD, NOT HELD, and no label by the cover, by 2.5, by the three profiles, by the two forms of F; P each word and each rule of no label | a reader that returns one word; a filter that changes a clean field or misses the added wave; a soft rule that misses a soft piece; a budget that does not cover the two-speed error |
| C14, the backgrounds | the three base states as loaded | the loop's center, normal and radius equal to those of the blocks of M5.49 to 10⁻⁶, q on the pieces equal to the blocks to seven digits, the piece geometry identical | another state, or another geometry, than the one F was computed on |

**Order.** Stage 0, one process: C1, C14, the prediction and its hashes (C5), C13, C2, C3, then Ga0 with C8 and C12 on it. The long jobs start only when stage 0 has met all of these. C4, C6, C7, C9 and C11 are read as their jobs end; C8, C10 and C12 per job.

**Tries and costs.** The try cap of each control is three. Between tries only the input of a control or a defect of a script may change, never a threshold of §§ 3, 5 and 6 nor a setting of §§ 1 and 2; each change is logged in the run log. What a failure costs: C1, C2, C3, or C8 or C12 on Ga0, no job past stage 0; C5, C13 or C14, no label on any grid; C4, C6 or C7, no label on any grid; C11 in a bin at a tone, every piece of that bin at that tone not read, on every grid; C9, no label at the frequency that failed; C8 on a job, no label from that job; C10 or C12 on a grid and a frequency, no label there.

### 6. The readings, frozen

Labels are given on grids b, d and e. Every entry (reading, line, edge, grid) carries its own word; the word of a reading is HELD when at least one entry is HELD and none is NOT HELD, NOT HELD when any entry is (the entry named), and no label when no entry carries a word. The words are tabled per grid, so the number of grids behind each word is on the page.

**W1, the falsifier, against the registration.** On line A at ω = 4, inside the near-threshold domain of the axis (§ 3: s = 1.75 to 4.75 on b, 1.75 to 4.25 on d and e), the statement of the read is:

| Statement | Rule |
| --- | --- |
| WAVE | at least nine pieces in ten of the domain read, and on each of the three profiles a run of consecutive domain pieces with the profile at or above 0 whose Φ, the run extended to its crossings and cut at the centers of the first and last pieces of the domain, is at least π; and the purity clause below holds on every piece of the run of the profile y + ε, the widest of the three |
| NO WAVE | at least nine pieces in ten of the domain read, and every read piece of the domain below 0 on each of the three profiles (y + ε < 0) |
| ZONE | at least nine pieces in ten of the domain read, a run of domain pieces at or above 0 on the profile y, and neither statement above (printed with its Φ on the three profiles) |
| undecided | anything else, a cover under nine in ten included |

**The purity clause** (a measured guard on the one word that would count against the registration): on a piece, the power of the lock-in amplitude outside W, $\sum_x ( \lVert A_2 \rVert^2 - \lvert a \rvert^2 )$ over its nodes in D with the inner product of M5.47 § 3, is under its power on W, $\sum_x \lvert a \rvert^2$. A soft wave puts little of its power on W (M5.47 § 11: 0.02 to 19 percent on the unit, median 0.6), so a soft share that matters in the projection shows as power outside W; a mid wave carries most of its power on W. A run that meets every other clause of WAVE and fails this one is ZONE, printed with the ratio of each piece. The clause does not enter NO WAVE, P or the edges, and its bound of 1 is the plan's choice: it catches a soft share above about 8 percent of the projection for the median soft wave and misses one whose own share on W is near the 19 percent end.

Nine pieces in ten of a domain of six or seven pieces means every piece of it. The statement must also be the same on window 1; otherwise it is undecided. Word: NOT HELD on WAVE, HELD on NO WAVE, no label on ZONE or undecided. Why π: half a wavelength, the least phase over which a standing pattern changes sign once, and the BAND rule of M5.49 (a run as long as its half wavelength) written as a phase. Φ is in the units of the author's law, whose speed is 1 (the plan's choice), which is stricter than M5.49's half wavelength by 1/c, about 1.08 on the axis: on grid b, the run that M5.49 read carries 3.05 radians in these units and about 3.3 in M5.49's. The author's wording gives the statement ("a propagating (phase-advancing) wave along the axis"); this plan adds the domain where it is read, what propagating means on a piece (y above 0 by its budget), how much phase makes a wave (π), and what the opposite statement needs (every read piece evanescent beyond its budget).

**W2, the axis against F.** F's statement on the same domain at ω = 4 by the same rule on y_F(1), with ε = 0.01 ω²; the ray form must give the same statement, or F's statement lies inside its margin. HELD when F and the read both say WAVE or both say NO WAVE; NOT HELD when both make one of those statements and they differ; no label otherwise. No label, known before the run, from § 4: on b, y_F(1) − ε is positive on one domain piece only and y_F(1) + ε on three, with Φ about 0.6 on y_F(1), so F's statement is ZONE; on d y_F(1) + ε reaches +0.08, so it is undecided; on e the pieces say NO WAVE (largest y_F(1) of −0.38) while the ray form has y_F + ε above 0 near s = 2.25. W2 is tabled for the profile.

**R2 to R4 and N2 to N4, the turning radii.** The entries and rules of M5.49 § 5, with the read of this task: at ω = 4 the inner and outer edges on PF and PD (R2 against the edges of F; N2 with r₀ = 2.77 and 6.31); at ω = 5.5 on A (R3 against F; N3 with r₀ = 0.81 and 8.38) and on PF and PD (R4 against F; N4 with r₀ = 1.56 and 9.74). For an edge expected at r₀, the window is the set of pieces with centers from 0.9 r₀ − Δ to 1.1 r₀ + Δ, covered when every one of them is read (any status). HELD: covered, and an edge of the same kind from 0.9 r₀ to 1.1 r₀ on all three profiles. NOT HELD: covered, and no such edge on all three. No label: not covered; r₀ under 2.5; the three profiles disagree; for an R entry also the two forms of F more than 3 percent of the radius apart or only one of them with the edge. The nearest read edge is always printed. The near-threshold domain does not bound these windows, which lie within about 0.1 r₀ + Δ of their edge. Known before the run: no inner edge carries a word (under 2.5, or next to the loop); 8.38 and 9.74 can carry one on grid e alone; 6.31 on PF and PD of d and e and on PD of b.

**P1 to P3, the phase across a band: this plan's extension of F.** The registration lists the local wavenumber near the turning points among what it does not predict, so P is not a test of the registration; it tests the author's rule on the background of OpenWave, carried from the edges to the band between them. On a line at ω, the stretch 𝒮 is the longest run of joined read pieces of the job inside the near-threshold domain that contains the outer edge of F on the pieces. Φ_read = Φ[y] and Φ_F(λ) = Φ[y_F(λ)] on that same 𝒮. HELD when $0.9 \Phi_F(1) \le \Phi_{\mathrm{read}} \le 1.1 \Phi_F(0.87)$ on all three profiles; NOT HELD when it lies outside on all three; no label when Φ_F(1) is under 1 radian, when Φ_F(0.87) exceeds 2 Φ_F(1) (the offset alone would double the prediction), when the three profiles disagree, or when the outer edge of F lies beyond 𝒮. The 10 percent is the registration's tolerance on an edge, carried to the phase (the plan's choice). P1: ω = 4 on PF and PD; P2: ω = 5.5 on A; P3: ω = 5.5 on PF and PD. At plan time, on the stretches that the read pieces of M5.49 give inside the domain, the intervals are:

| Entry | Φ_F(1) | Φ_F(0.935) | Φ_F(0.87) | Interval of HELD |
| --- | --- | --- | --- | --- |
| P1, b PF | 2.45 | 3.21 | 3.73 | 2.20 to 4.10 |
| P1, b PD | 3.50 | 4.74 | 5.89 | 3.15 to 6.48 |
| P1, d PF | 2.65 | 3.79 | 4.67 | 2.38 to 5.14 |
| P1, d PD | 3.69 | 4.94 | 6.02 | 3.32 to 6.62 |
| P1, e PF | 2.74 | 3.89 | 4.73 | 2.47 to 5.20 |
| P1, e PD | 2.46 | 3.63 | 4.56 | 2.22 to 5.02 |
| P2, e A | 20.31 | 26.78 | 33.00 | 18.28 to 36.30 |
| P3, d PD | 16.34 | 19.19 | 21.16 | 14.71 to 23.28 |
| P3, e PF | 14.92 | 19.06 | 22.63 | 13.43 to 24.90 |
| P3, e PD | 14.39 | 18.46 | 22.27 | 12.95 to 24.49 |

No entry is withheld at plan time by the rule of 1 radian or by the factor of two; the stretches of the run may differ from these.

**Registered before the run.** Expectations, which no script uses. (1) W1: the cutoffs that M5.49 read on the axis lie between the two thresholds of § 4 on every grid (on the domain of grid b, 0.92 to 1.00 of F), so ZONE or undecided and no label on b, d and e, with b the nearest to WAVE (Φ of 3.05 by M5.49's cutoffs, 3.35 at λ = 0.935). W1 reads HELD only if the cutoff measured at ω itself lies on or above F over the domain (λ of 0.993 or more on e, 1.003 on d, 1.021 on b), and NOT HELD only if it lies under F by about 6.5 to 9.5 percent and the run also meets the purity clause. The one tracked record of the polarization on these pieces is the free noise of M5.49 (`outside_W` at the target 4.0 in [`m5_49_reads.json`](../data/m5_49_reads.json)): on the axis pieces that M5.49 read under ω = 4 the share of the power outside W is 0.54 to 0.78 on b, 0.87 to 0.94 on d and 0.87 to 0.88 on e, against 0.07 to 0.59 in the plane band. If the driven field is no purer than that noise, no axis piece meets the clause and WAVE is not reachable: W1 is then a one-sided read, HELD or no label. A source along W may give a purer field; that is not known before the run. (2) R2 to R4 and N2 to N4 as in M5.49: HELD where covered, the read edges 3 to 8 percent outside F. (3) P1 to P3 inside their intervals, near the middle value λ = 0.935 (3.2 to 4.9 for P1, 26.8 for P2, 18.5 to 19.2 for P3). (4) C9, C10 and C12 met; C11 met if the ratio holds on the unit as § 3 says. Nothing in this section sets an icon: the plan reports, and the Klein-Gordon cell is re-read later by the repo owner, by a ruling.

### 7. Blindspot pass

| Unknown | Guard |
| --- | --- |
| A steady state in a closed band, where the phase is not a wavenumber | the read is the local law, not the phase (Scope); the uniform damping; C10 and C12 |
| The damping changes what is read | the real part of the law is unchanged and the ratio removes the speed (§ 3); C9 in the vacuum, C11 at half the damping on the unit; the binned term $b_\gamma$ of ε |
| Two speeds on the unit make the ratio inexact, more so deep in a band | stated in § 3 with its size; the damping term binned by $\lvert y \rvert / (\omega \gamma)$; C11 per bin; C13 (b) shows the budget covers the made-up case |
| The mid wave out of reach, the soft waves on W | the near-threshold domain for W1 and P; the soft rule and the c² bound; the purity clause on the word WAVE of W1; limit (6) states what is left unguarded (a soft share of roughly 10 to 60 percent); C13 (b) for the soft rule; the deep vacuum cases of C9 tabled |
| The source next to a read | the rule on the raw g; the filter term only where the filtered g is under 10⁻⁵; C12 on the read nodes; C9 near the threshold |
| The background moves by itself | f₀ subtracted; C3 (wiring) and C4 |
| Two tones, terms of higher order | C6; the exact orthogonality of the windows, C13 (a) |
| Transients of the ramp | the settling; two windows, C10; C12 |
| The step | C7 |
| The read is not blind | limit (4); the prediction hashed before any driven job; the expectations written from M5.49 before the run |
| A yes or no at its threshold | W1 asks, inside the near-threshold domain, Φ of at least π on all three profiles or every piece evanescent beyond its budget, and the same on two windows; the continuous profile of y with ε is the result, the word only its summary; the two thresholds in λ are stated in § 4 |
| The offset of M5.49 | an interval decided now, from the least piece ratio the record holds, never a point; P withholds where the interval alone decides |
| The literal phase read as the falsifier | a descriptor only, with the reason (Scope, fact 3) |
| P taken for a test of the registration | named as this plan's extension of F, with the registration's own line on what it does not predict |
| This plan's choices taken for the author's | named as the plan's in the findings: the uniform damping in place of the layer; the subtracted residual force; the source's place, polarization, width, ramp and two tones; the ratio estimator and the budget; the soft rule and the near-threshold domain; the meaning of propagating and the π of W1 in units of speed 1; the interval of λ; the tolerance of P (10 percent); the floors and exclusions |
| An unvalidated instrument | C9 and C13 give every reader a known answer, with both answers near the threshold at each frequency; C12 ties the fields to the linearized equations; C1 to C3 tie the driver to the stepper of M5.47 |
| An artifact read as a result | the windows, the damping, the filter (where allowed) and the vacuum each enter ε; C7 for the step |
| An unfalsifiable check | each control states its failing input; C3 is named a wiring check; P gives no label where the interval spans a factor of two; C9 asks both answers |
| Resolution mistaken for accuracy | ε beside every y; numbers printed with their budget |
| Ceiling-to-point creep | λ = 0.87 enters only as an end of an interval; the expectations carry no word |
| A diagnostic outside its regime | the regime table of § 3 |

### 8. Consumers, parallel shape, estimate, not computed

**Consumers.** The words W1, R2 to R4, N2 to N4 and P1 to P3 are read by the repo owner's post to the author on discussion #186: they need C1 to C14 met and the two audits. The Klein-Gordon cell of `MODELS.md` reads them only through a later ruling of the repo owner. The driver of § 1 and the ratio estimator of § 3 are reusable by a driven read on another action (the action that M5.51 ports): they need C1 to C3, C8, C9 and C12 again on that action.

**Parallel shape.** One single-threaded process per job, the thread variables set to 1 before NumPy loads; the jobs are independent of each other, and inside a job the stepper makes one pass over the nodes per stage (M5.47). Stage 0 is serial, since it gates the launch; C1 runs its two short runs of E1 beside it. When stage 0 has met its controls, Ge starts first: it sets the clock, 3356 steps on 105³, about 2.2 times the cost per step of 81³ by the node count. Then Gd, GbG, Gb, VD1 to VD4, GaH, GaS, GaP and GaN, in that order, as slots free. The ceiling of twelve single-threaded processes holds for this task and the sister task together: while the sister task has chains running or queued, this task holds at most four long processes and the sister task at most eight (stage 0 with the two runs of C1 counts as three); when the queue of either task is empty, the other may take the freed slots, the two together never above twelve. The reads of a grid run as one more process when its job ends, counted in the ceiling. The GPU is not used: the action needs f64.

**Estimate.** Machine: thirteen driven jobs (three on the labeled grids, one more on grid b, four in the vacuum, five on grid a, GaN being the twin), set by the clock of Ge, with the caps of § 2; the reads and the plots are a desk pass. Sub-agents: the builder on Opus (one); a numbers check on Sonnet; a record audit on Opus in a fresh context. The user's touchpoints: the go, shared with M5.51, and the review. Autonomy: unattended from the go to the review; the scope cut on a cap of Ge is decided above, so no ruling is needed in between.

**Deviations.** Anything forced off this plan takes the conservative option (a word withheld rather than given) and is logged in this document as it happens, with what it changes.

**Not computed.** A front read before its first return; the literal phase as a word; the decay beyond an outer edge as a word (a descriptor only); a frequency scan; the author's layer and a source at the loop's center; a steady state without loss; the inner edges; the pattern of the wave in three dimensions; the light and the soft polarizations; the action of the author's later post; another time part; another spacing and the limit at zero spacing; the bias of a cutoff on a background that is not uniform; an error bar beyond ε; grids a and c as labeled grids.

### 9. The vacuum control checked at plan time

An exact linear steady state of the vacuum jobs was solved at plan time with code written for this plan (the held shell as a sine transform of the lattice law of M5.47 § 1 with the damping, the source of § 1 decomposed on the six branches of the coded Hessian; W carries 0.9999 on the mid branch, 0.0084 on the light branch and nothing on the rotations), and read with the estimator of § 3 on the 26 read pieces of A, PF and PD. It is a made-up vacuum, not data of the task. Near the threshold and deep in the propagating range the read is exact to the tolerances of C9 with a margin of fifteen or more; in the deep evanescent cases the light wave and the tail of the source take over the projection, and the soft rule alone does not withhold the pieces.

| Case | y | Read pieces | Removed by the soft rule | Worst error of $m^2_{\mathrm{read}}$ | Tolerance | $c^2_{\mathrm{read}}$ |
| --- | --- | --- | --- | --- | --- | --- |
| VD2, ω = 4 | +0.96 | 26 | 0 | 0.0004 | 0.16 | 1.000 |
| VD3, ω = 4 | −0.96 | 26 | 0 | 0.0017 | 0.16 | 1.000 to 1.001 |
| VD1, ω = 5.5 | −3.02 | 26 | 0 | 0.0098 | 0.30 | 1.000 to 1.006 |
| VD4, ω = 5.5 | +1.82 | 26 | 0 | 0.0007 | 0.30 | 1.000 |
| VD2, ω = 5.5 | +15.21 | 26 | 0 | 0.0063 | 0.30 | 1.000 |
| VD3, ω = 5.5 | +13.29 | 26 | 0 | 0.0107 | 0.30 | 1.000 to 1.001 |
| VD1, ω = 4 (tabled) | −17.3 | 12 (14 outside the c² bound) | 0 | 18.5 (median 4.1), all read EVAN | none | 1.04 to 2.11 |
| VD4, ω = 4 (tabled) | −12.4 | 22 (4 outside the c² bound) | 0 | 10.8 (median 2.2), all read EVAN | none | 0.56 to 1.89 |

On the two deep cases the mid branch read alone is off by up to 1.44 and 0.54, from the tail of the source (g up to 3.3 × 10⁻⁴ on piece nodes) in a field that evanescent; the mixture with the light wave does the rest. Their pieces lie outside the near-threshold domain, which is why no word reads them. The near-threshold evanescent cases (VD3 at ω = 4, VD1 at ω = 5.5) are read exactly on every piece, so C9 needs no restriction by distance from the source.

**Plan audit (2026-11-23, before any driven job).** An independent pass in a fresh context derived the driven equations, the steady state and the estimator again, solved the steady state of the vacuum jobs exactly with its own code and read it with the plan's estimator, and recomputed the plan-time numbers; it ran no driven job and opened no wave array. It confirmed the driven equations, H′ and its energy identity, the steady-state equation and its signs, the ratio and the sign of Im Z, the lock-in windows, the step, the settling and the decay factors, the wavenumbers of fact 3, the weights of the source and of the filter, the settings of the vacuum jobs, q on the circles, the exclusion by the source, the axis numbers and the quotations. It changed, by item: the read is unfiltered and the vacuum control sits near each threshold, the filtered read having failed on the widened source; the soft rule was added, with the limit it states; W1 has one cover domain near the threshold for both of its words; the decay beyond an edge was dropped as a word; the interval of the offset was widened to what the record holds; the damping term of the budget is binned; and P is named as this plan's extension of F. The drafter and the auditor ran on the same model in separate contexts, the session's model having read the draft in full and each revision beside them; the vacuum table of § 9 was recomputed by the drafter with its own code after the audit. A second pass of the same auditor on the revised plan solved the vacuum jobs again against the control as rewritten (every bound met by a correct reader with a margin of 28 times or more, none by construction) and found: that the purity clause, added by the session after the first pass, is not met by the free noise of M5.49 on any axis piece, which expectation (1) now states; the P3 rows computed over the whole line instead of the stretch, corrected; the bounds of C11 resting on an input whose recipe was not given, now given, with a failure of C11 costing its bin alone; and the budget of C9's statuses, now stated.

## FINDINGS (2026-11-23)

> The run of the plan above, made on 2026-11-23; the sections below are filled as the run proceeds.

### 10. The controls

| Control | Tries | Word | What was read |
| --- | --- | --- | --- |
| C1, the stepper | 1 | ✅ met | `m5_47_evolve.py --controls` unchanged, its output written to the local checkpoints: 97 leaves of [`m5_47_controls_stepper.json`](../data/m5_47_controls_stepper.json) reproduced bit for bit (timing and series left out), K0 to K5 passed |
| C2, the driver reduces to the stepper | 1 | ✅ met | from rest on B0a, a_s = 0, γ = 0, f₀ not subtracted: M and p equal to `RK4.step` after each of 50 steps (`numpy.array_equal`); the state moved by 1.45 × 10⁻⁷ |
| C3, the hold (a wiring check) | 1 | ✅ met | largest $\lvert M - M_0 \rvert$ after 500 steps: 0 |
| C4, the subtraction against a twin | 1 | ✅ met | GaP − GaN against Ga0, window 2, both tones, on the 15779 nodes of D of grid a with g under 10⁻³: worst 1.57 × 10⁻⁵ (bound 0.01) |
| C5, the prediction first | 1 | ✅ met | [`m5_52_drive_predicted.json`](../data/m5_52_drive_predicted.json) written by the stage-0 process, its block and file hashes written to the local run log, before Ga0 started in the same process; the hashes of blocks b, d, e and V2 of M5.49 matched; every read took the five block hashes and matched them before it opened a lock-in amplitude |
| C6, linearity | 1 | ✅ met | Ga0 against GaH, both tones and windows: worst 1.65 × 10⁻⁷ (bound 0.02) |
| C7, the step | 1 | ✅ met | Ga0 against GaS: worst 5.0 × 10⁻⁵ (bound 0.01) |
| C8, the energy | 1 | ⚠️ met on every job but GaN | ratio of the largest residual to the largest $\lvert H'(t) - H'(0) \rvert$: Ga0 8.5 × 10⁻⁶, GaH 2.6 × 10⁻⁵, GaS 6.3 × 10⁻⁶, GaP 7.6 × 10⁻⁶, VD1 1.7 × 10⁻⁵, VD2 2.4 × 10⁻⁶, VD3 1.3 × 10⁻⁷, VD4 1.5 × 10⁻⁵, Gb 1.6 × 10⁻⁵, GbG 5.1 × 10⁻⁶. ❌ GaN 0.0918 (bound 0.02): with no source and f₀ not subtracted, its largest $\lvert H'(t) - H'(0) \rvert$ is 1.50 × 10⁻¹¹ and its residual 1.38 × 10⁻¹², the size of the residual of the other jobs of grid a (1.6 to 2.1 × 10⁻¹²). Cost by § 5: no label from GaN, which carries none |
| C9, the known answer | 1 | ⚠️ met at ω = 4, not met at ω = 5.5 | ω = 4, VD2 and VD3: 26 of 26 pieces read, none soft, worst $\lvert m^2_{\mathrm{read}} - m^2 \rvert$ 0.0014 and 0.0033 (tolerance 0.16), $c^2_{\mathrm{read}}$ within 0.10 and 0.52 % of 1, every status as set, no edge, Φ within 0.05 % of $\sqrt{y}$ times the stretch. ω = 5.5: VD4, VD2 and VD3 meet every clause; ❌ VD1 (y = −3.03) fails one clause, $c^2_{\mathrm{read}}$ within 1 % of 1: it reads 1.0244 (PF), 1.0239 (PD) and 1.0234 (A) on the innermost read piece (s = 0.75, where the field is smallest), falling toward 1 outward, while every other clause holds (26 read, none soft, worst mass error 0.050 against 0.30, every piece EVAN, no edge). Window 1 reads 0.92 to 0.93 on the same pieces. $b_{\mathrm{vac}}$ = 0.0033 at ω = 4 and 0.050 at 5.5. Cost by § 5: no label at ω = 5.5 |
| C10, the windows | 1 | ✅ met on the jobs read | median of $\lvert y_i - y_i^{(1)} \rvert$: Gb 0.0044 (ω = 4) and 0.0045 (5.5), GbG 0.0025 and 0.0092, grid-a jobs 0.0046 and 0.010, against 0.16 and 0.30; the deep evanescent vacuum cases (VD1 and VD4 at ω = 4, tabled) read 1.64 and 0.88 |
| C11, the damping | 1 | ❌ not met at ω = 4; met at 5.5 | 90th percentile of $\lvert y(\mathrm{Gb}) - y(\mathrm{GbG}) \rvert$ over the read pieces of grid b. ω = 4: bin I 0.351 over 8 pieces (bound 0.16), bin II 2.22 over 5 (bound 0.96), bin III 3.07 over 1 (bound 1.92). ω = 5.5: bin I no piece, bin II 0.48 over 3 (bound 1.82), bin III 3.44 over 13 (bound 3.63). On the same pieces $c^2_{\mathrm{read}}$ itself moves with the damping (for one, the axis at s = 1.75: 0.87 at γ = 0.3, 1.42 at 0.15). Cost by § 5: at ω = 4 every piece of bins I, II and III is not read on every grid |
| C12, the steady state | 1 | ⚠️ met on every job but GaN | worst ratio: Ga0, GaH and GaS 3.7 × 10⁻⁴, GaP 5.5 × 10⁻⁴, VD1 2.1 × 10⁻³, VD2 3.0 × 10⁻⁴, VD3 7.1 × 10⁻⁴, VD4 4.9 × 10⁻³, Gb 1.8 × 10⁻⁴, GbG 1.1 × 10⁻⁴ (bound 0.03). ❌ GaN 758: no source, so its lock-in holds the drift of the base state, not a driven steady state; grid a carries no label |
| C13, the readers on made-up inputs | 1, and a second run on the final reads script | ✅ met | (a) amplitudes to 3.3 × 10⁻¹⁵, the 5.5 tone at 4 under 3.3 × 10⁻¹⁵, the tone at 4.3 against the analytic response 1.6 × 10⁻¹⁵. (b) every plane wave to 10⁻¹⁴ with the right status, filtered unchanged, the added wave read within 0.07 of 14 and 18 filtered and 1.6 to 1.85 off unfiltered, the soft input NOT READ (soft); the two-speed input over ε on at most 9.97 % of the draws of a bin (bin I at ω = 4), the 90th percentiles of the error 0.14 to 2.44 at ω = 4 and 0.19 to 4.26 at 5.5. (c) 24 of 24 outcomes |
| C14, the backgrounds | 1 | ✅ met | loop center equal, normal within 3.3 × 10⁻²⁴, radius within 6.1 × 10⁻⁸ of the blocks of M5.49 on b, d and e; piece geometry identical; q on the pieces equal to seven digits |

**What the controls leave.** By the costs of § 5, C9 at ω = 5.5 withholds every entry at that frequency (R3, R4, N3, N4, P2, P3), and C11 at ω = 4 removes every read piece of bins I to III at that frequency on every grid, which leaves the entries at ω = 4 (W1, W2, R2, N2, P1) without read pieces. Both follow from jobs of grid b and of the vacuum alone; they hold whatever grids d and e read.

### 11. The jobs

| Job | Grid | Steps | Capped | C8 |
| --- | --- | --- | --- | --- |
| Ga0, GaH, GaP | a, 33³ | 3356 of 3356 | no | met |
| GaN | a, 33³ | 3356 of 3356 | no | not met (round-off, above) |
| GaS | a, 33³ | 6712 of 6712 | no | met |
| VD1, VD2, VD3, VD4 | vacuum, 49³ | 3356 of 3356 | no | met |
| Gb | b, 65³ | 3356 of 3356 | no | met |
| GbG | b, 65³ | 5456 of 5456 | no | met |
| Gd | d, 81³ | 3356 of 3356 | no | met (1.8 × 10⁻⁵); its field not yet read (deviation 2) |
| Ge | e, 105³ | see below | | |

No job had a node where the kinetic matrix was not positive.

### Deviations from the plan (logged as they happen)

1. **Code tests before stage 0.** Before stage 0 the scripts were tested: the pure readers of C13 on their made-up inputs, a trial prediction written to a scratch file outside the repository (it reproduced the plan-time numbers of § 4 and the P table of § 6), and 41 steps of the driver on grid a written to scratch (no lock-in window reached, no amplitude written). Stage 0 then ran as one process in the plan's order. No threshold, setting or rule changed.
2. **The reads of grids d and e delayed by an instruction of the orchestrator.** While Gd and Ge ran, the orchestrator ordered that no read, lock-in inspection, C12, words or plot be made on the fields of Gd and Ge until a second, separate scoring rule for those fields had been frozen, so that it is frozen before anyone has read them. The jobs ran untouched. The reads of d and e were delayed, not changed: no rule, job or target of this plan changed.
3. **Choices the plan left open, fixed by the scripts.** The near-threshold domain of a line holds the pieces that M5.49 reads (F on the pieces is defined on those). A piece is excluded by the source when any of its nodes, in D or not, has g at 10⁻³ or more. The term $b_\gamma$ pools the read pieces of all four lines of grid b. Where window 1 does not read a piece that window 2 reads, the window term of ε is infinite (the piece can only be MARGIN). C9's "no edge" is checked on the three profiles. The reference Φ_F of the prediction file (§ 4) is taken on the stretches of the pieces M5.49 reads; it reproduced every number of § 4 and of the P table of § 6. In C11 a bin with fewer than five pieces is still scored on its own 90th percentile, while its budget term takes the largest of the bins as § 3 fixes.
