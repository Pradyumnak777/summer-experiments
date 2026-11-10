# `score` vs `score_detected`

Per-frame metrics from `custom_metric.py`. Each frame is pushed along the SVM normal 3 steps down and 3 steps up, and every push is compared only to that frame's own recon (never to a neighbouring push).

- Down push: correct if `I(push) < I(recon)`
- Up push: correct if `I(push) > I(recon)`

Both metrics count the same correct pushes. The only difference is what they divide by.

## Example

One frame, recon intensity **0.80**:

| Push | Intensity | Compared to recon 0.80 | Outcome |
|---|---|---|---|
| -6 | 0.60 | lower, should be lower | correct |
| -4 | no mask | can't compare | no measurement |
| -2 | 0.90 | higher, should be lower | wrong |
| +2 | 0.85 | higher, should be higher | correct |
| +4 | no mask | can't compare | no measurement |
| +6 | 0.88 | higher, should be higher | correct |

3 correct, 1 wrong, 2 with no mask.

| Metric | Divides by | Value |
|---|---|---|
| `score` | all 6 pushes (no mask = wrong) | 3 / 6 = **0.50** |
| `score_detected` | only pushes with a mask (no mask = left out) | 3 / 4 = **0.75** |

Exam analogy: 6 questions, 3 right, 1 wrong, 2 blank. `score` marks the blanks wrong (3/6), `score_detected` grades only what was attempted (3/4).

If no push is missing a mask, the two are identical.

## Reading the values

- 1 is the best, 0 is the worst.
- Chance level for `score_detected` is 0.5. For `score` it is lower when masks go missing, roughly 0.5 x the fraction of pushes that have a mask.
- `score` tends to be pessimistic, `score_detected` optimistic. Read `score_detected` alongside the `no burst found %` row.

## Missing masks

| Where the mask is missing | `score` | `score_detected` | `spearman` |
|---|---|---|---|
| Recon | frame discarded | frame discarded | frame discarded |
| Some pushes | frame kept, those pushes count as wrong | frame kept, those pushes are left out | frame discarded |
| All 6 pushes | frame kept, score is 0 | frame discarded | frame discarded |



#NOTE: spearman coeff outputs nan if ANY of the 7 frames(recon + perturb) do not have a mask. So the the final mean spearman is reported only on frames that contaainb all masks.