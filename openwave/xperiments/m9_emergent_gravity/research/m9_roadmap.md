# M9 / NSM (EMERGENT GRAVITY), ROADMAP

> Column roadmap. Index of work only; results live in the task
> documents. Live: [IN PROGRESS](#in-progress). Queued: [BACKLOG](#backlog).
> Closed: [DONE](#done). Reading rules: [§ CONVENTIONS](#conventions).
>
> **Archive pointer (2026-08-22).** The campaign records this document cites (M9.3 to M9.73: notes, scripts, data, latex, APPLICATION) are not in the tree; they are parked in PR #441's branch history at `a5640709` and return as per-task PRs against registered roadmap rows. Their links below were reduced to plain text at the merge trim.

## IN PROGRESS

> Nothing running.

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |

## BACKLOG

> Native gravity cell after the algebra gate. Headless only. Pre-registered.
> Parked rows: work the author's agent produced before the column was registered; the records live in PR #441's branch history (`a5640709`) and each returns as its own per-task PR, at which point the row moves.

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |
| 🚧 M9.3 | Metric phenomena note | Parked. Domain note: Einstein+Lambda, FGHMV cited, dS [O], torsion out of scope. Record at `a5640709` | author per-task PR |
| 🚧 M9.4 | Axial UV deformation | Parked. Q4b tree-level r(k)=(3/16)/(1+k^2/M^2). Record at `a5640709` | author per-task PR |
| 🚧 M9.5 | Q4a holographic pair | Parked. Selection-uniqueness negative; existence still [O]. Record at `a5640709` | author per-task PR |
| 🚧 M9.6 | de Sitter FGHMV bar | Parked. Copy obstructed (sign, isometries). Record at `a5640709` | author per-task PR |
| 🚧 M9.7 | Jacobson substitute | Parked. Not [P]. Record at `a5640709` | author per-task PR |
| 🚧 M9.8+ | Entanglement-gravity campaign | Parked. Papers 14 to 73 (A1/A2 ansatz, first law, Gauss and enclosed-energy results, vacuum-energy negatives); the author registers each as a task row before its PR. Record at `a5640709` | author per-task PR |

## STATUS AT A GLANCE (2026-09-02)

| Question | Answer |
| --- | --- |
| Where is M9? | Registered column (admission: discussion #442, 2026-08-15; activation merge: PR #441). MODELS.md column present, Hehl-Datta certification at ⚠️, every other cell 🚧 |
| What kind of column? | Gravity-certification EFT. Matter installed. Closer to M8 than M5 |
| What decides the first cell? | An in-platform gravity result against a pre-registered gate, landed by a per-task PR. M9.2 C1 passed only as inherited Newton, C2 failed, so the Newton cell stays 🚧 |
| Is the column staffed? | No. The collaborator invitation was issued twice and expired unaccepted, and the author last posted 2026-08-20, so the column is registered and dormant. The registered slice stands on maintainer reruns of both gates, so no result here waits on contact |

## CONVENTIONS

> Standing reading rules. Dated events live in [§ CHANGE-LOG](#change-log).

**Spec of record.** [`m9_theory_canonical.md`](m9_theory_canonical.md) wins when
documents disagree. Application: [discussion #442](https://github.com/openwave-labs/openwave/discussions/442).

**Mode of work.** Headless first. No launcher and no GUI until a gravity cell
exists.

**Standing rules.** Pre-register before a run. Adversarial audit on substantive
claims ([`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) § 1). Author-facing notes
follow [`METHOD_NOTE.md`](../../../../dev_docs/METHOD_NOTE.md).

**Campaign scripts are not official task IDs.** `m9_2_ib_hadamard.py` and
neighbors record \(I_B\) / axial negatives. Official M9.2 is Newton only.

## DONE

| TaskID | Title | Description | Completed |
| --- | --- | --- | --- |
| ✅ [M9.1](tasks/m9_1_task_details.md) | Hehl-Datta elimination | Closed 2026-08-15. Gate PASS r=3/16 both signatures (scatter <1e-15). Paper spin dual C2 FAIL (measured -1/2 not -1/4). Full record: [note](findings/m9_1_hehl_datta_note.md) + [task](tasks/m9_1_task_details.md) | 2026-08-15 |
| ✅ [M9.2](tasks/m9_2_task_details.md) | Newton limit | ⚠️ Closed 2026-08-15. C1 PASS, C2 FAIL (Dirichlet images). Inherited Einstein, not GEM; the MODELS.md Newton cell stays 🚧. [note](findings/m9_2_newton_note.md) | 2026-08-15 |

## CHANGE-LOG

2026-09-02. Column dormant. The collaborator invitation was issued at
admission and re-issued at activation; both expired unaccepted, so the
author holds no repository role. No author activity since the 2026-08-20
comment on PR #441. Backlog rows stay parked and return as per-task PRs
whenever the author resumes. Nothing is withdrawn and nothing is waiting
on the maintainer side.

2026-08-22. Activation merge (PR #441) trimmed to the onboarding slice:
scaffold, M9.1, M9.2. Campaign records M9.3 to M9.73, latex and the
application body parked at `a5640709`, registered above as BACKLOG rows.
MODELS.md column added (tag NSM). Official ID M9 since 2026-08-15.

2026-08-15. Draft column opened from the author repo
[n4hy/New_Model_Emergent_Gravity](https://github.com/n4hy/New_Model_Emergent_Gravity).
M9.1 locked as the certification gate. Official model ID pending maintainer
assignment. No papers copied into `theory/`.

2026-08-15. M9.1 closed on the algebra gate. Closed \(I_B\) and axial-EC
attempts recorded as briefing negatives. M9.2 Newton gates locked with no
run. M9.3 metric domain note written; MODELS.md untouched. Application
body posted as discussion 442; PR 441 is the first package.

2026-08-15. Q4 split. Selected UV (Q4a) not invented. Unique quadratic
axial deformation (Q4b) constructed, tree-level matched to 3/16, audited.
Final Status UV label for Q4a unchanged.

2026-08-15. Q4a answered as selection: uniqueness negative, existence
still open. No CFT invented. Paper 16.

2026-08-15. Q2 decided at the FGHMV bar: copy onto the cosmological
horizon is obstructed (sign + isometries). No dual invented. Paper 17.

2026-08-15. Jacobson is not a [P] substitute for Q2. Paper 18.

2026-08-15. Cosmological questions are metric only. Torsion removed
from the Q2 bar. de Sitter is the spinless vacuum (Paper 19). FGHMV
copy remains obstructed.

2026-08-15. Derived the cosmological minus sign from SdS
(T dS + dM = 0). Paper 20. Not FGHMV in dS.

2026-08-15. A2 then A1 on a 1d free fermion. A2 not refuted
(locked R ratio). A1 UV α independent of IR mass. Paper 21. Not 4d.

2026-08-15. Required A2 on 2d/3d Dirac: C2 pass. 1d scalar
instrument rejected. Paper 22.

2026-08-15. A2 on a \(3+1\)D diamond waist (geodesic ball), two
spacings at matched \(mL\). C2 pass; auditor CONFIRMED. Not
\(a\to 0\) and not a de Sitter selection. Paper 23.

2026-08-15. A1 on the same waist: area coefficient \(\alpha=0.245\),
UV drift \(\le 4.3\%\). Unsubtracted sea \(\varepsilon\simeq -1\)
is not mean-zero curvature. Paper 24.

2026-08-15. CHM shape of the modular hop on the diamond:
\(\rho=-0.987\), \(R_{\mathrm{shape}}=0.162\). Auditor CONFIRMED.
Not \(1/4G\). Paper 25.

2026-08-15. First law: \(\delta S\) tracks a local kernel
(\(\rho=0.85\)). CHM envelope loses to a flat hop kernel (C3
FAIL, auditor CONFIRMED). Paper 26.

2026-08-15. Horizon split. Bulk C3 fail. Surface C3: CHM beats
flat (\(N=10\) and auditor \(N=12\)). Tracking floors fail.
Paper 27.

2026-08-15. \(S^3\) curvature axis: Ricci \(6/\rho^2\), Wick
sign flip, Haar mean zero. The \(S^3\) picture is a guess.
Paper 28.

2026-08-15. Horizon C3 vs linear: radial weight beats flat at
\(R=4\), \(n_{\mathrm{flip}}=0\). Parabola tied with \(R-r\).
Paper 29.

2026-08-15. \(R=5\) first-law attempt: \(25/30\) occupancy
flips at \(\varepsilon=0.002\). Instrument rejected. Not
Planck. Paper 30.

2026-08-15. C4 scored at \(R=5\) by half-filling: TIE
(\(n=30\), \(n_{\mathrm{flip}}=0\)). Tracking collapses.
Paper 31.

2026-08-15. Bloch vs CHM on every dimer in the ball
(\(1302+624\)). Bloch \(\lvert\rho\rvert\sim 0.03\), CHM
\(\rho\sim -0.99\). Guess closed as a description of \(K\).
Paper 32.

2026-08-15. Fixed \(H\), deform the region. Cubes miss the ball
area law (RMS \(5.67\) vs \(0.27\)). Same \(A_{\mathrm{cut}}\),
different \(S\). Paper 33.

2026-08-15. \(\delta S\) of \(512\) balls is linear in a
Gaussian source (\(\rho_{\varepsilon}=1\),
\(\rho_{\mathrm{CHM}}=-0.986\)). Kernel not unique. Paper 34.

2026-08-15. Point source: flat/enclosed energy beats CHM
(\(\rho_{\mathrm{flat}}=-0.83\) vs \(-0.47\)). Auditor
REFUTED C2. Paper 35.

2026-08-15. 1d CHM calibration of that instrument: flat
still wins (\(\rho_{\mathrm{CHM}}=-0.054\)). \(\delta S\)
tracks \(\mathrm{Tr}(K_{\mathrm{mid}}\Delta C)\), not the
vacuum first law. Paper 35 is not a 3d no-go. Paper 36.

2026-08-15. Fixed-\(H\) occupation transfer: vacuum first
law holds. 1d still selects enclosed energy. 3d balls
select CHM (auditor CONFIRMED). Not Einstein. Paper 37.

2026-08-15. Author guess (still a guess): need a shape,
not a sphere. Cubes prefer a cube-native weight; exported
CHM is not robust (auditor REFUTED C2e). Paper 38.

2026-08-15. Geometric half: hop conformal \(\delta S\) tracks
\(\delta A\) (\(\rho=0.918\)) but \(\eta\) is not constant
(rel IQR \(3.17\)). Not Clausius. Not Einstein. Paper 39.

2026-08-15. M9.2 Newton lock run. C1 attractive \(1/r^2\)
PASS. C2 isolated \(\Phi\) FAIL (Dirichlet box). Inherited.
No MODELS.md. Paper 40.

2026-08-15. Proper area \(1/t^2\) is linearly the same
Clausius test as Paper 39's length. \(\eta\) still
scatters (rel IQR \(3.17\)). Paper 41.

2026-08-15. Hop-area vs CHM energy: \(\rho=0.996\).
Area is the same conformal bump. Matter-only first law
has \(\delta A=0\). Paper 42.

2026-08-15. Cut-correlator area moves at fixed \(H\), is
not \(S\), tracks \(P_{\mathrm{CHM}}\) at \(\rho=-0.925\).
Still not Clausius. Paper 43.

2026-08-15. Two masses: all-ball CHM confirmed; both-inside
Gauss not killed (auditor REFUTED C2b). Paper 44.

2026-08-15. \(R=3\), \(69\) both-inside balls: flat beats
CHM (\(R=0.053<0.298\)). Auditor REFUTED C2b. Paper 45.

2026-08-15. Gauss first-law constant \(\kappa\approx 0.97\),
rel shift \(1.7\%\) one vs two masses. Auditor CONFIRMED.
Paper 46.

2026-08-15. \(\kappa\) weighs enclosed mass and locates the
source (exact on \(N=12\); auditor off by one site).
Paper 47.

2026-08-15. Pipeline \(\delta S\to M_{\mathrm{hat}}\to\)
inherited DST Poisson C1 PASS (\(n=65\) residuals
\(\le 3.0\%\), slope \(-2.037\)). Auditor CONFIRMED.
Entanglement supplies the mass; Einstein still does
\(1/r^2\). Not a derived Poisson. Paper 48.

2026-08-15. Real \(\delta e\) as \(\rho\): far-field Newton
sees \(\sum\delta e\), not \(M_{\mathrm{hat}}\) (C_hat
FAIL \(41\)--\(43\%\)). Packet is compact; the \(28\%\)
gap is offset-ball bookkeeping. Auditor CONFIRMED.
Paper 49.

2026-08-15. Two real packets, no \(M_{\mathrm{hat}}\):
one enclosing ball reads the pair (\(0.84\%\)); midpoint
cancels (\(4\times 10^{-5}\)); exterior matches two-point
Coulomb (\(\le 0.8\%\)). Auditor CONFIRMED. Linear
Poisson is not a discovery. Paper 50.

2026-08-15. Wide vs compact: first law grows \(16\times\)
on an extended source and plateaus on a star. Nested
volume vs area withdrawn (collinear). Linear \(a\)
FAIL --- not Newtonian \(\Lambda\). Paper 51.

2026-08-15. Unequal masses: interior null is \(M/r^2\)
(\(0.0025L\)), not the CM (\(0.16L\) away). Auditor
side CONFIRMED, precision REFUTED. Paper 52.

2026-08-15. Diamond waist, staggered \(m\): \(\kappa\)
moves \(0.14\%\) at \(m=0.25\) and \(0.60\%\) at
\(m=0.50\). Auditor CONFIRMED (\(0.21\%\)). Not a
massless-cube artifact. Paper 53.

2026-08-15. Periodic band-edge transfer: \(\delta e\)
flat to \(10^{-12}\). Volume first law on \(N=12\).
Inherited \(a\propto r\) (slope \(1.029\)). Auditor
volume law REFUTED on wrapping balls. Paper 54.

2026-08-15. First-law mass plus Gauss, no DST:
star slope \(-1.997\), sea slope \(+1.266\).
Auditor CONFIRMED. Paper 55.

2026-08-15. \(\sigma\)-scan: Gauss slope \(-1.87,-0.90,+0.40,+1.00\),
sea \(+1.27\). All inward. Paper 54 ``Newtonian
\(\Lambda\)'' withdrawn. Dust, not de Sitter. Paper 56.

2026-08-15. Fermi-sea vacuum: \(E_{\mathrm{vac}}<0\)
(cutoff, \(N\)-dependent). \(S\) tracks area, not
volume. \(a_{\mathrm{try}}=-S/R^2\) slope \(0.26\).
Not \(\Lambda\). Paper 57.

2026-08-15. First law tracks \(\delta e\)
(\(\rho=0.99999\)), not raw \(e\) (\(\rho=-0.64\)).
Vacuum energy is \(2762\times\) the packet.
Sea subtracted, not repulsive Newton. Paper 58.

2026-08-15. Complement of an enclosing ball:
\(\delta S(B^c)\sim 10^{-5}\) vs \(\delta S(B)=0.196\).
Not \(T\mathrm{d}S+\mathrm{d}M=0\). Paper 59.

2026-08-15. \(\kappa\) vs \(\alpha\): rel range \(0.57\).
Tracks \(2h(\alpha)/(\alpha\Delta E)\) at
\(\rho=0.999998\). Paper 46 universality is
fixed-\(\alpha\). Reusable mass is \(\sum\delta e\).
Paper 60.

2026-08-15. Gauss from \(\sum\delta e\), no \(\kappa\):
star \(-1.9979\), sea \(+0.967\), \(\alpha\)-spread
\(10^{-13}\). Auditor CONFIRMED. Paper 61.

2026-08-15. 216 balls, three \(\alpha\): \(P_{\mathrm{flat}}\)
wins \(\delta S\) every time. CHM and \(T_K\) fall as
\(\alpha\) grows. No flip. Paper 62.

2026-08-15. Enclosure identity: \(f_S=\delta S/S_{\mathrm{global}}\)
tracks \(f_E=P_{\mathrm{flat}}/M_{\mathrm{global}}\)
(\(\rho=0.991,0.996,0.999\)). Centered ball recovers
\(f\approx 1\). Source-inside C_enc FAIL (leak).
Auditor \(\rho=0.998\). Paper 63.

2026-08-15. Sea transfer, slabs break \(A\)/\(V\):
grow \(2.59,2.55,2.51\) (volume \(3\), area \(1\)).
\(\rho(f_S,f_E)=0.994\). Auditor grow \(2.56,2.47\).
Paper 63 is not a packet artifact. Paper 64.

2026-08-15. Two-term \(\delta S=aV+bA\): in-sample
rel RMS \(0.08\), same as volume-only. Held-out
rods miss by \(5.2\). \(b\) flips sign. Not a
horizon piece. Paper 65.

2026-08-15. Hop virial stress: sea \(r=0\), star
\(0.163\). C_lambda FAIL. Vacuum \(P/E=-0.88\)
is not a source. Paper 66.

2026-08-16. Thermal \(\mu=0\) scan: \(N=12\) \(r(T)\)
runs \(-58\to +0.07\) and crosses \(-1\) at \(T=0.5\).
Auditor \(N=10\) all \(r>0\). Not \(\Lambda\).
Paper 67.

2026-08-16. Flux law \(g=-M/A\): star \(-1.9998\),
sea \(+0.992\). Hop \(E_{\mathrm{int}}\) overlap at
\(d=2\) (\(2.8\%\)), not \(1/r\). Paper 68.

2026-08-16. Pair direction \(\hat n=\nabla_c M_{AB}\):
median \(10.75^\circ\) to \(M/r^2\), \(62^\circ\) to
CM. Auditor direction CONFIRMED; axis-null
REFUTED as under-resolved. Paper 69.

2026-08-16. Exact 1d open-hop basis, \(dps=80\),
no LAPACK: angle \(10.746^\circ\) unchanged.
Additivity \(6\times 10^{-8}\) is pair ortho.
Auditor \(N=11\) all CONFIRMED. Paper 70.

2026-08-16. Pair \(\mathbf g\) superposition: mass
maps add (\(2\times 10^{-7}\)); \(\mathbf g\) residual
\(0.255\) FAIL. Far angle \(0.008^\circ\). Add
masses, then one \(\nabla M\). Paper 71.

2026-08-16. Pair \(S=2h(\alpha_A)+2h(\alpha_B)\)
exact. Enclosure \(\rho=0.957\). Both-inside leak.
\(\nabla\cdot g\) not Poisson (auditor REFUTED).
Paper 72.

2026-08-16. Pair \(R=5\) midpoint: \(f_S=0.991\),
\(f_E=0.988\). Axial offset leaks (\(0.886\)).
Auditor perpendicular offset CONFIRMED. Paper 73.
