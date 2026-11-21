# M10.3 · Free persistence of a prepared pair

We propose Interval Echo as the single M10 executable. This task concerns row 4. The M10.2 signed-source candidate at `c330c6e7f23506b5f87c339cd73906928c1a7cd9` supplied a compact ray, a transfer spectrum and a static read. Its experiments left free lifetime unresolved. Interval Echo uses a complete 99-state coherent seam memory, local load exchanges, an endpoint contrast at coupling 8 and degree compensation on the spherical federation. All seams act during the free interval. The detector at the launch cell can record an occupied seam or quiet. Each appointment uses a fresh preparation. The earlier pin and its static results retain their own scope. [M10.2 review](../findings/m10_2_maintainer_review.md) · [Interval Echo law and author evidence](https://github.com/FloatingPragma/observer-patch-holography/tree/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo)

## Reproduction gates recorded before platform execution

This is a reproduction of disclosed author results. Binding, local probabilities, the derivative window bound, and the source choice were known when the terms were fixed. The author had frozen the cell-13 experiment locally before its acquisition. The public matrix states the simplest criterion as persistence of a particle state in free evolution. Its M6 entry also asks for evolution of a neutral profile. This test concerns a prepared bound pair with unassigned charge and spin. It supports no electron identity or cosmological lifetime. [Official criterion](https://github.com/openwave-labs/openwave/blob/26fbb5fdd07049625d6bb7930af2f6bf545fb6fc/MODELS.md#L104)

1. Run the included sparse-exponential program with the canonical 20-cell board, then compare every saved address byte, all counts, and the rounded probabilities. The historical cell-0 result at exposure 150 must reproduce the printed 0.983881 and 0.149559 within 0.000002.
2. In cell 13, start with unit loads at 156 and 157, all collars blank. Count local occupied-seam reports from 4,096 terminal reads at each of exposures 1, 7 and 37 in both arms. Include quiet. Use the frozen per-setting PCG64 seeds in the replay. The lower bounds for the native fraction must exceed 0.80 at all three settings, and the exchange-only upper bound must be below 0.65 at the last. The simultaneous one-sided Hoeffding radius is approximately 0.02857. This changes only the binding term.
3. The separate Taylor propagation and energy-variance interpolation must reproduce a downward-rounded lower bound of at least 94% on the home-cell seam probability throughout [0,37]. This is a statement about terminal probabilities over an ensemble.
4. Replacing the native contrast by zero must make the native persistence gate reject with exit 1. The detector must give the correct sharp outcomes on two local pairs and a separated pair. A mismatch in a saved receipt, or a mutation rejected only by the file matcher, is insufficient.
5. Verify the source export and replay board digests and compare both sector generators to the pinned law. The author checker tests the full binary terminal instrument in the 99-state seam space. Any material source or gate change calls for a fresh review. Retain failed outcomes.

Obtain the author checkout at the proposed pin. Set `ECHO_INPUTS` to the absolute path of its `openwave/interval_echo/row_04` directory. All data inputs are read offline from that checkout under its terms. This platform task includes the Apache-2.0 replay software. From the platform repository root, with Python 3.12, NumPy 2.3.5 and SciPy 1.18.1, run

```sh
cd openwave/xperiments/m10_oph/research/scripts/m10_3
python -B check_inputs.py "$ECHO_INPUTS"
python -B row_04_check.py --board "$ECHO_INPUTS/FEDERATION_LEVEL0.npz" --receipt "$ECHO_INPUTS/row_04_receipt.json"
python -B row_04_window.py --inputs "$ECHO_INPUTS" --board "$ECHO_INPUTS/FEDERATION_LEVEL0.npz"
python -B row_04_check.py --board "$ECHO_INPUTS/FEDERATION_LEVEL0.npz" --receipt "$ECHO_INPUTS/row_04_receipt.json" --remove-contrast
```

The last command is expected to exit 1 at the persistence gate. The 20-cell exposure and its units are supplied. No additional interventions occur in flight. The author carries out the platform replay after the planning commit. A pass supports a request for full row 4 credit. The maintainers decide the mark and pin. [Complete design and raw inputs](https://github.com/FloatingPragma/observer-patch-holography/tree/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo)
