# M4.11 – N as the Internal Lock-In Point: Drift, Resolution, Grouping

## Status
DONE (post-hoc)

## Objective
Verify that $N_{\text{geom}} = 778.8025$ is the internal lock-in point of the
Enhanced EWT model. For each sector — $G$, $\alpha$, lepton AMMs, and
mixing angles — we compute the value at $N_{\text{geom}}$ and measure how
much it drifts when $N$ is changed.

The analysis is performed at three levels:

1. **Drift** — how quickly each sector departs from its value at $N_{\text{geom}}$.
2. **Resolution** — how narrowly each sector can locate $N$, computed as
   its relative error divided by its logarithmic derivative
   $d\ln O / d\ln N$.
3. **Grouping** — which sectors actually constrain $N$, which are only
   consistent, and which provide no constraint at all.

The reference is $N_{\text{geom}}$ itself, not a fit to CODATA/PDG. Every
sector has by construction zero drift at $N = N_{\text{geom}}$. The
question is how sharply the model reacts when $N$ moves away from that
point.

Only PDG quark masses are used for the Cabibbo sector. No EWT quark masses
enter the calculation.

## Method

1. Compute the reference value of each sector at $N_{\text{geom}}$.
2. Scan $N$ from 500 to 1100 with step 2.
3. For each $N$, compute all sector values using the same functions as
   the M4.7 emergence engine:
   - $G(N)$ via the full self-consistent chain,
   - $\alpha^{-1}(N)$,
   - $a_e(N), a_\mu(N), a_\tau(N)$,
   - $\sin^2\theta_W(N)$ and $\sin\theta_C(N)$.
4. Compute the relative drift
   $|O(N) - O(N_{\text{geom}})| / |O(N_{\text{geom}})|$.
5. Compute the logarithmic derivative
   $\beta = d\ln O / d\ln N$ at $N_{\text{geom}}$ by central difference.
6. Compute the resolution as
   $\text{relative error} / |\beta|$.
7. Find the separately optimal $N$ for each sector by log-space scan and
   bisection.
8. Group the sectors by resolution:
   - **PRIMARY** — resolution < 1%,
   - **WEAK** — resolution between 1% and 100%,
   - **NO CONSTRAINT** — resolution > 100%.

## Results

### Reference values at $N_{\text{geom}}$

| Quantity | Value |
|----------|-------|
| $\alpha^{-1}$ | 137.036262364181 |
| $G$ | $6.677519975508460 \times 10^{-11}$ |
| $a_e$ | $1.159916228104 \times 10^{-3}$ |
| $a_\mu$ | $1.166212608460 \times 10^{-3}$ |
| $a_\tau$ | $1.176838130442 \times 10^{-3}$ |
| $\sin^2\theta_W$ | 0.232758852461 |
| $\sin\theta_C$ | 0.224287669401 |

### Section 1: Drift relative to $N_{\text{geom}}$

| Sector | drift at $N_{\text{geom}}$ | drift at $-10$ | drift at $+10$ |
|--------|-------------------------------|------------------|------------------|
| $\alpha^{-1}$ | $3.117 \times 10^{-10}$ | $4.251 \times 10^{-9}$ | $3.527 \times 10^{-9}$ |
| $G$ | $1.245 \times 10^{-2}$ | $1.825 \times 10^{-1}$ | $1.314 \times 10^{-1}$ |
| $a_e$ | $1.326 \times 10^{-6}$ | $1.808 \times 10^{-5}$ | $1.500 \times 10^{-5}$ |
| $a_\mu$ | $1.460 \times 10^{-6}$ | $1.990 \times 10^{-5}$ | $1.652 \times 10^{-5}$ |
| $a_\tau$ | $2.272 \times 10^{-4}$ | $3.093 \times 10^{-3}$ | $2.575 \times 10^{-3}$ |
| $\sin^2\theta_W$ | $4.720 \times 10^{-5}$ | $6.434 \times 10^{-4}$ | $5.341 \times 10^{-4}$ |
| $\sin\theta_C$ | $9.202 \times 10^{-6}$ | $1.255 \times 10^{-4}$ | $1.041 \times 10^{-4}$ |

The drift at $N_{\text{geom}}$ is not exactly zero because the scan grid
step is 2, and the nearest grid point to $N_{\text{geom}} = 778.8025$ is
$N = 778$. The physically meaningful values are the drifts at $\pm 10$,
which are unaffected by this discretization.

### Section 2: Resolution

| Sector | $\beta = d\ln O / d\ln N$ | relative error | resolution | $N_{\text{opt}}$ |
|--------|------------------------------|----------------|------------|-------------------|
| $\alpha^{-1}$ | $+0.000000$ | $1.921 \times 10^{-6}$ | $6.353 \times 10^{+0}$ | 105.85 |
| $G$ | $-12.000002$ | $4.817 \times 10^{-4}$ | $4.014 \times 10^{-5}$ | 778.83 |
| $a_e$ | $+0.001285$ | $2.277 \times 10^{-4}$ | $1.771 \times 10^{-1}$ | 661.63 |
| $a_\mu$ | $+0.001415$ | $2.504 \times 10^{-4}$ | $1.770 \times 10^{-1}$ | 661.69 |
| $a_\tau$ | $+0.220297$ | $3.159 \times 10^{-4}$ | $1.434 \times 10^{-3}$ | 779.92 |
| $\sin^2\theta_W$ | $-0.045754$ | $6.655 \times 10^{-3}$ | $1.455 \times 10^{-1}$ | 910.04 |
| $\sin\theta_C$ | $-0.008921$ | $5.497 \times 10^{-5}$ | $6.162 \times 10^{-3}$ | 774.03 |

The exponent for $G$ is exactly $-12$, not $-3$. This is a consequence
of the self-consistent derivation of $\lambda_l$ in the v5.0.0 chain:
$\lambda_l \propto N^{-6}$, and $G$ inherits an additional $N^{-9}$
dependence through $N_{\nu,\text{eff}}$, compounding with the explicit
$N^{-3}$ factor.

### Section 3: Grouping

| Group | Sectors | Resolution range |
|-------|---------|------------------|
| PRIMARY (resolves $N$) | $G$, $a_\tau$, $\sin\theta_C$ | $< 1\%$ |
| WEAK (consistent only) | $a_e$, $a_\mu$, $\sin^2\theta_W$ | $1\% - 100\%$ |
| NO CONSTRAINT | $\alpha^{-1}$ | $> 100\%$ |

The primary agreement between the two strongest independent sectors is:

$$
G \text{ vs } a_\tau: \quad N_{\text{opt}} = 778.8338 \text{ vs } 779.9210,
\qquad \Delta = 0.1396\% \text{ of } N_{\text{geom}}.
$$

This is the central quantitative result: **$G$ and $a_\tau$ independently
locate $N$ with resolutions of 0.004% and 0.14%, and they agree to
0.14%.**

## Interpretation

1. **Gravity is the primary leg of the lock-in.**  
   The drift of $G$ at $N_{\text{geom}} \pm 10$ exceeds 10%, and its
   resolution is $4 \times 10^{-5}$. The very high exponent $N^{-12}$
   makes gravity by far the most sensitive probe of the lattice stiffness.

2. **The tau AMM is the only other sector that resolves $N$.**  
   With $\beta_{a_\tau} = 0.22$, $a_\tau$ locates $N$ to 0.14%.
   Crucially, its construction is completely different from $G$: it comes
   from the recursive lepton shell damping, not from the gravitational
   chain. The agreement between $G$ and $a_\tau$ at the 0.14% level is
   the falsifiable core of the shared-lattice claim.

3. **Electron and muon AMMs are weak confirmations.**  
   Their projection $\mathcal{O}_\mu = 1/(4\pi^2)$ suppresses the
   sensitivity of the muon shell, and the electron AMM is dominated by the
   nearly $N$-independent Schwinger term. Their resolutions of ~18% mean
   they are consistent with $N_{\text{geom}}$ but cannot discriminate it
   from neighbouring values.

4. **The Cabibbo angle is conditionally primary.**  
   $\sin\theta_C$ formally lands in the PRIMARY group with a resolution
   of 0.62%. However, this result is conditioned on the central PDG values
   for the light quark masses. The experimental uncertainty on
   $m_d$ and $m_s$ spans about 14%, roughly fifteen times larger than
   the geometric correction being tested. At the low corner of that range
   no positive $N$ reproduces $\sin\theta_C$. The Cabibbo leg must
   therefore be treated as conditional, not as a fully independent
   constraint.

5. **The Weinberg angle is weak and scheme-sensitive.**  
   The logarithmic derivative is small ($-0.046$), giving a resolution
   of 14.6%. Moreover, the comparison between the model's
   $\sin^2\theta_W = 0.23276$ and the PDG value uses the MS-bar scheme,
   while the on-shell value from the PDG masses is $0.22305$. The
   geometric correction $C_{\text{gap}}$ sits in the same slot as a known
   scheme-conversion factor of 1.06%, so the Weinberg leg cannot yet carry
   independent weight.

6. **Alpha provides no constraint on $N$.**  
   With $\beta \approx 3 \times 10^{-7}$ and a resolution of 635%,
   $\alpha$ cannot serve as a lock-in test. This is expected: $A_\pi$
   dominates $\alpha^{-1}$, and the $\epsilon_M$ correction is tiny.

7. **Gravity as a residual force explains the hierarchy.**  
   The extreme sensitivity of $G$ to $N$ is not an accident. In the
   Enhanced EWT framework gravity is a residual force, arising from the
   near-exact cancellation of large geometric factors. Small changes in the
   lattice parameter therefore produce large relative changes in $G$,
   while first-order quantities such as $\alpha$ remain nearly unchanged.
   This is a feature of the hierarchy, not a defect of the model.

## Technical notes

- The scan used a step of 2 in $N$, so the nearest grid point to
  $N_{\text{geom}}$ is $N = 778$. This gives a small nonzero drift at
  the reported “$N_{\text{geom}}$” row for very sensitive sectors such as
  $G$. The drift values at $\pm 10$ are unaffected by this
  discretization and are the physically meaningful measure.
- The logarithmic derivatives $\beta$ were computed by central difference
  in $\log N$ with step $10^{-6}$.
- The separately optimal $N$ values were found by log-space scan and
  bisection, not by local extrapolation, because the functions are not pure
  power laws over the scanned range.

## Artifacts

- `m4_11_lock_in_n_scan.py`

## Reference

Enhanced EWT manuscript, version 5.0.0:
DOI: 10.5281/zenodo.22540635