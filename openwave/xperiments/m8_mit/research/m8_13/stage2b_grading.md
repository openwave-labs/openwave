# Stage 2b: grading the author's argument

Your stage-1 and 2a returns are committed and will not change. Besides these instructions, you now receive three files: the author's argument, `S0_S3_MAXIMUM.md`; its checker, `check_s3_maximum.py`; and the checker's log, `check_s3_maximum_log.txt`. Running the author's checker and getting the author's output is not a verification of anything.

## How to grade

- **ESTABLISHED**: the stated route holds as written.
- **ESTABLISHED, SUPPLIED**: the stated route omits a case or a derivation, and you supply it, shown. Name exactly what you supplied.
- **GAP**: the step cannot be established, as written or with anything you can supply, and you say what is missing.
- **DEFECT**: a false statement, or a necessary hypothesis left unstated, even if the result can be rescued.

Grade the argument, not its conclusion. A true conclusion does not make a step ESTABLISHED: if the text's reasoning does not reach it, the grade is ESTABLISHED, SUPPLIED, with what you added, or GAP. GAP is strict: a step you can complete is never a GAP. Complete it and name what you supplied. The same line runs the other way: a derivation you write out is a supplied part, even when the text contains its ingredients, so it is ESTABLISHED, SUPPLIED and never ESTABLISHED as written. Every number you assert comes from code you wrote.

## What to grade

Give one verdict per step, and for step 3 one per part:

1. Step 1, the invariant form.
2. Step 2, the two bounds and their equality cases.
3. Step 3, in five parts:
   - **3a.** The reduction to `λ_max(e₁fₓ² + e₂f_y² + e₃f_z²)` over unit traceless `e`, and the split into a zero eigenvalue and three 2×2 blocks.
   - **3b.** Each block's bound, one verdict per block.
   - **3c.** The equality set: the points of the constraint ellipse at which some block reaches `15/√6`. Say whether every block's equality points are found, and how you know.
   - **3d.** The passage from `‖Q‖ = 15/√6` to a state lying in the top eigenspace of the operator for an extremal `e`.
   - **3e.** That eigenspace, as `span{|3,3⟩ₙ, |3,−3⟩ₙ}`.
4. Step 4, the combination of the three equality cases.

Then reconcile: does the author's equality set agree with your stage-1 answer? Report any disagreement in either direction.
