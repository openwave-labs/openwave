# M10.4 · Free persistence of a prepared pair on a connected board

$$
h_\kappa=U G_\kappa U^\dagger,\qquad
G_\kappa=0_{\rm blank}\oplus\left[\tfrac12(I-\mathcal R)+\kappa d^2\right]\otimes I_{\rm loads}.
$$

Each Interval Echo seam has nine endpoint load words and eleven memory states. The write $W$ stores a fair coherent coin and the signed load difference $d$, splits the total load equally, and uses the coin to place any odd unit. The echo $U$ has blank/record blocks $(0,W^\dagger;W,I-WW^\dagger)$. The reversal $\mathcal R$ changes the sign of $d$. On blank memory, $h_\kappa$ acts as $(I-\mathrm{SWAP})/2+\kappa(n_i-n_j)^2$, with dimensionless contrast strength $\kappa$ in exchange units. Echoing two empty ends writes the equal coin superposition with $d=0$. Only equal-load writes have overlap with that label. The inverse echo therefore maps a neutral record to blank or zero-contrast support, both annihilated by $G_\kappa$. A written contact has zero exchange and contrast for every subsequent endpoint word.

**Scope**: On one 20-cell, 240-port board, the calculation certifies the probability that cell 13 detects a pair on a home seam at any supplied free exposure, relative to eleven persistent engineered contacts. Translation on a uniform background, a physical lifetime, electron identity, charge, mass and spin are outside this measurement.

## Preparation and terminal measurement

The two unit loads start on $(a,b)=(157,156)$ with all other ports empty. Among $b$'s neighbors, $c=166$ is the unique nonadjacent tip distinct from $a$ with the same original degree as $a$ (six). The tips share $b$ and $u=163$. Write the eleven edges leaving $\lbrace a,b,c\rbrace$ except $au$ and $cu$. Four addressed sharp swaps move the vacant port along $c,b,a,b,c$ so each write sees two empty ends and the loads return to $ab$. Two writes, $(152,166)$ and $(157,170)$, lie on intercell seams. All 240 ports remain connected.

In the ordered basis $X=ab,\ Y=ac,\ Z=bc$, acquisition enables only $ab$, $bc$ and the degree reference $6-\deg(b)=1$ at $b$. The two live contacts count $(1,2,1)$ unequal endpoint words. Adding the reference on $X,Z$ gives the same contrast on all three words.

```math
H_{\rm src}=K_{\rm src}+2\kappa I,\qquad
2K_{\rm src}=\begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix}.
```

After each pulse query occupation of $b$ by echo, pointer read and inverse echo on a neutral contact with its other end empty. Let $Q$ project onto $X,Z$, and write $D=(X-Z)/\sqrt2$, $B=(X+Z)/\sqrt2$. The exchange eigenvalues are $0,1/2,3/2$. The common contrast phase drops out. On $D,B$ the occupied branch $Q\exp(-i\tau K_{\rm src})Q$ has amplitudes $e^{-i\tau/2}$ and $(2+e^{-3i\tau/2})/3$. At $\tau=2\pi/3$ the bright amplitude is $1/3$. Three occupied reads give $v=(-13X+14Z)/27$ from $X$, up to a common phase. A first vacancy gives $Y$ with probabilities $4/9,\ 4/81,\ 4/729$ at the successive reads. Swap $bc$ to reset $Y$ to $X$ before a retry. The eight-attempt budget uses at most 24 pulses and queries and seven retry swaps.

$$
f=364/729,\qquad \rho=\left(\sum_{j=0}^7 f^j\right)vv^\dagger,\qquad
p_{\rm exhausted}=f^8=1-\operatorname{tr}\rho .
$$

During flight enable the entire native generator with no interventions. On all $\binom{240}{2}=28{,}680$ distinct-port configurations let $L$ be the graph Laplacian for single-load moves across blank contacts into empty ports, and $m_{ij}$ the number of such moves from $ij$. The degree references use the original board degrees.

$$
H_\kappa=K+\kappa C,\qquad K=L/2,\qquad
C_{ij,ij}=m_{ij}+(6-\deg i)+(6-\deg j).
$$

Each of $X,Z$ has two moves, into $Y$ and $bu$. Their off-diagonal amplitudes $-1/2$ cancel in $D$, their exchange diagonals equal one, and both contrasts equal three. Hence $H_\kappa D=(1+3\kappa)D$ on the full board. A fresh preparation undergoes one terminal read. The echo, copied classical pair flag and inverse echo detect binary occupancy on each of the 30 geometric home seams, including written seams. With $U_t=\exp(-itH_\kappa)$ the address law is $p_s=\langle s|U_t\rho U_t^\dagger|s\rangle$. Home probability is $\sum_s p_s$ and quiet is its complement, including exhaustion. The home projector contains $D$. For $R$ reads per attempt and $M$ attempts this gives

$$
p_{\rm home}(\kappa,t)\ \ge\ \frac{1-f_R^M}{1+9^{-R}},\qquad
f_R=(1-9^{-R})/2,\qquad 1-p_{\rm home}\ \le\ 9^{-R}+2^{-M}.
$$

## Equation-to-code map

| Equation or operation | Pinned implementation |
| --- | --- |
| Echo, blank and neutral columns, binary flag and neutral writes | [`echo`, `generator_parts`, `seam_term` · law.py:67–97][law] · [`certify_native_collar`, `seal_path` · corridor_pair.py:49–121][collar] |
| Source and flight generators, integer eigenvector and bright escape | [`geometry`, `source_parts`, `full_parts`, `dark_certificate`, `bright_escape` · open_pair.py:24–135][flight] |
| Acquired density, floor and 31-bin instrument | [`monitored_source` · corridor_release.py:31–47][monitor] · [`source_bound`, `source`, `terminal` · open_pair_release.py:19–65][release] · [`read_release` · open_pair.py:138–173][readout] |
| Independent propagation and receipt comparison | [`board_generator`, `flight_amplitudes`, `stochastic_population` · open_pair_check.py:28–129][check] · [`verify` · open_pair_check.py:178–197][verify] |

## Results and controls

For $R=3$ and $M=8$, exhaustion is 0.0038635883. Conditional on success the dark fraction is $729/730$. **Across all enrolled preparations the exact lower bound exceeds 0.994771841 for every real $\kappa$ and every $t\ge0$.** It clears the known 0.94 persistence criterion algebraically. The port geometry, cadence, dark-state identity and threshold were known in choosing the preparation. Time is dimensionless exposure in inverse exchange units. The appointments use a supplied time parameter without an observer clock reading. [Connected protocol][protocol]

| $\kappa$ at $t=37$ | Acquired, coherent | Acquired, port-dephased | Acquired, random swaps | Sharp $X$, written contacts |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.995196058 | 0.652907216 | 0.123196093 | 0.655439565 |
| 0.5 | 0.995945068 | 0.926296116 | 0.123196093 | 0.929888823 |
| 8 | 0.996109886 | 0.986454527 | 0.123196093 | 0.990280564 |

Dephasing removes the off-diagonal elements of $\rho$ once before coherent flight. Independent rate-$1/2$ swaps on the same live contacts evolve the same acquired populations by $\exp(-tK)$, giving home probability 0.753088868 at $t=1$, below the 0.94 gate. Both controls retain exhaustion. Sharp $X$ omits the filter. Under coherent flight the symmetric $B$ has probability $t^8/18432+O(t^{10})$ of reaching a different cell. Thus the available exits carry loads. [Full bin probabilities][records]

**Matched blank-memory control.** Keep $\rho$, exhaustion, $\kappa=0.5$, $t=37$ and the detector fixed, and restore all eleven contacts to blank for a counterfactual flight. Home probability falls from **0.995945068 to 0.440929105**. The historical M10.3 sharp source on the same blank board gives 0.416819359. At $t=37$ that sharp source gives 0.980272 at 8 and about 0.984 at $-8$. M10.3 inherited 8 from a design sweep testing strengths from 0 to 128, with 8 and 128 as headline values. The archive gives no selection of 8 independent of persistence. The all-strength result belongs to the prepared memory environment. [Original coupling sweep][sweep] · [M10.3 decision][review]

The nine settings $(\kappa,t)\in\lbrace0,0.5,8\rbrace\times\lbrace1,7,37\rbrace$ give 1,116 address and quiet probabilities across four arms, rounded to trillionths. M10.3 drew 4,096 simulated records per setting for its sampling gates (native lower bounds above 0.80 and final control upper bound below 0.65). A raw-board Chebyshev and Poisson replay of M10.4 differs from its archived bins by less than $1.8\times10^{-11}$ (tolerance $2\times10^{-10}$). [M10.3 sampling design][sampling] · [Independent replay][check]

**Audit.** All 30 connected-pair and acquisition tests pass at the pin. An independent check summed the signed exchange amplitudes from $D$ at every destination outside $X,Z$. Their squared norm is zero with every write. Omitting a single write gives $1/8$ for eight contacts and $1/4$ for three. This tests exact cancellation. The finite-window probabilities of the eleven omissions require separate evolution. [Native-column and changed-contact tests][tests] · [Source and denominator tests][source-tests]

## Reproduce

Start with the [source and flight operators][flight]. The two data inputs from that OPH revision are the [240-port board][board] and [ordered probability archive][records]. From the OpenWave research scripts directory, run the following with their paths. The script checks both SHA-256 digests, computes the marked and blank flights, and compares each marked bin. [OpenWave replay](../scripts/m10_4_connected_pair.py) · [Developer's run summary](../data/m10_4_reproduction.json)

```sh
python -I -B m10_4_connected_pair.py BOARD.npz --archive connected_records.json
```

For the separate author implementation, run `python -B -O -W error -m model.candidate_laws.interval_echo.openwave.row_04.open_pair_check --full` from the pinned OPH checkout.

[law]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/contrast_memory_spherical_v1/law.py#L67-L97
[collar]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/corridor_pair.py#L49-L121
[flight]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/open_pair.py#L24-L135
[monitor]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/corridor_release.py#L31-L47
[release]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/open_pair_release.py#L19-L65
[readout]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/open_pair.py#L138-L173
[check]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/open_pair_check.py#L28-L129
[verify]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/open_pair_check.py#L178-L197
[protocol]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/PROTOCOL.md#L22-L24
[records]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/connected_records.json
[board]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/executable_model/carriers/FEDERATION_LEVEL0.npz
[sweep]: https://github.com/FloatingPragma/observer-patch-holography/blob/972d81b9b6068938e52457b59497766499222b5d/model/polynomial_sv/SPEC.md#L92-L103
[review]: https://github.com/openwave-labs/openwave/pull/745#pullrequestreview-6005365373
[sampling]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/openwave/interval_echo/row_04/ROW_04_DESIGN.md#L5-L11
[tests]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/test_open_pair.py#L16-L85
[source-tests]: https://github.com/FloatingPragma/observer-patch-holography/blob/09f062f77f7f540f2783f128ae6314c5d8b818ea/model/candidate_laws/interval_echo/openwave/row_04/test_pair_acquisition.py#L10-L40
