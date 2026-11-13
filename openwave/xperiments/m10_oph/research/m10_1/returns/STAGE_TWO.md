# STAGE TWO

**C2: holds, weakly sharp.** The bound (Lagrange remainder of 1-cos t after t^6 is at most t^8/8!, times 12 terms, |u.n| <= 1) is in the source above the comparison and I re-derived it: valid. All 20 solver rows are inside (largest diff/bound 0.168). My run at 70 digits: 8 new directions ((2,-3,7), (5,1,1), (-4,6,3), (1,0,0), (1,1,1), (phi,1,0), (1,-1,0), (7,2,-9)) x a*k in {1/4..1/32} = 32 samples, all inside, diff/bound 0.079 to 0.140.

**C3: holds, exact.** On the solver's polynomials, every monomial has a-power 2m-2 and degree 2m. p(a/2) - 2^-(2m-2) p = 0 identically for degrees 2, 4, 6 (factors 1, 1/4, 1/16). Invariant coefficients scale 1/4, 1/16; I6 coefficient 1/16; ratio 16/75 unchanged.

**C4: holds, qualified.** Test and alterations were defined in source before use; removal and doubling are rejected, the original accepted. My own breaks, rotating one direction by an exact rational rotation (cos 3/5, sin 4/5) and replacing one by a coordinate axis: both rejected. Rotating all twelve: accepted. The test can pass a non-invariant operator: it checks degree 2 only, and six cube axes pass it while their degree-4 sum is not invariant (exact). It also hard-codes |v|^2 = 2+phi, so it is valid only for vectors of that norm.

**Mutations (red outcomes):**

| Mutation | Result |
| --- | --- |
| degree-6 qz^6 coefficient x (1+1e-3) in C2 | red in 3 of 32 samples, only at a*k = 1/16 and 1/32; green at 1/4, 1/8 |
| polynomials evaluated at a/2 in C2 | red in 32 of 32 |
| C3 with 1/8 instead of 1/16 at degree 6 | fails |

So C2 alone cannot certify coefficients to 1e-3 at large a*k; the exact C1 and C3 comparisons carry that.

**script findings:**
1. Duplicate `mp.mp.dps = 80` line, harmless.
2. Invariant coefficient uses the typed sphere average 1/(2m+1), not computed from the definition. It is correct, and the zero remainders at degrees 2 and 4 cross-check it.
3. The REPORT figures (0.17, 16/75) are prose; the values in `results.json` are computed.
4. No loose tolerance: strict `<=` against the stated bound, exact `== 0` elsewhere.
5. A copy of `solve.py` rerun in `scratch_rerun` exits 0 and writes a `results.json` identical in content to the shipped one.

