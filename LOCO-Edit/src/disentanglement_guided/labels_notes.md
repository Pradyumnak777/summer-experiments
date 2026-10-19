# Burst intensity labels in `vae_guided.py`

## Where the raw label comes from
`get_gene_burst_intensity` in `disentanglement_semantic_eval/gan_dci.py`:

```python
img = np.array(Image.open(chB_path).convert('L')).astype(np.float32) / 255.0
burst_int.append(img[burst].mean())
```

- `img`: the **chB** crop (the gene burst channel), scaled to [0, 1]
- `burst`: the hand-drawn burst mask (True/False per pixel)
- `img[burst].mean()`: mean chB brightness inside the mask, giving one number per frame
- only frames with a non-empty annotated mask are kept

**chA → chB fix:** the first version used chA, which is wrong. So the first guided-VAE run
(exc_mse ≈ 0.13, inh_mse ≈ 0.70) and `gan_burst_R_matrix` were trained/computed on the wrong labels
and need redoing. Don't reuse an old `burst_labels.npz` cache, because `vae_guided.py` loads it if it exists.

## Standardizing (z-score)
```python
y_mean, y_std = burst_int.mean(), burst_int.std() + 1e-8
labels = (burst_int - y_mean) / y_std
```

The label means: **how many standard deviations this burst is above or below the average burst.**

Example with the real stats (`y_mean = 0.7231`, `y_std = 0.1430`):

```
label = (0.7452 - 0.7231) / 0.1430 = 0.0222 / 0.1430 ≈ +0.155   -> slightly brighter than average
```

| raw burst_int | label | meaning |
|---|---|---|
| 0.0 (black) | −5.06 | lowest possible |
| 0.580 | −1 | one std dimmer than average |
| 0.723 | 0 | exactly average |
| 0.745 | +0.155 | slightly brighter |
| 0.866 | +1 | one std brighter |
| 1.0 (saturated) | +1.94 | highest possible |

- The order of bursts is unchanged; only the units change (pixel brightness → "stds from the average").
- The labels are **not** limited to [−1, 1]: they have mean 0 and std 1, and here the possible range
  is about [−5.06, +1.94] (lopsided because the average burst is already close to the max brightness).
- This differs from image normalization `(x − 0.5) / 0.5`. That one uses fixed numbers chosen to map
  [0, 1] exactly to [−1, 1] (range scaling). This one uses the data's real mean and std (centering and spread).
- `y_mean` and `y_std` are saved (`burst_labels.npz`, `heads_*.pt`), so predictions convert back with
  `real = pred * y_std + y_mean`.

## Why standardize: the MSE gets a fixed meaning
Labels and MSE are different things. A label can be −5, but the MSE of predicting standardized labels reads as:

- **MSE = 1**: no better than always guessing the average (0), meaning nothing was learned
- **MSE = 0**: perfect prediction
- in between: the fraction of variance left unexplained (e.g. 0.30 means 70% explained)

Without standardizing, always guessing 0.723 would give MSE ≈ 0.143² ≈ 0.02. That looks small but
means nothing was learned, and the scale would change with the data's spread.
This fixed meaning is why the inhibition term is capped with `clamp(inh_mse, max=1.0)`:
past 1, the other latents already carry no intensity information.

## Loss terms don't need a shared range
```
loss = recon + β·KL + EXC_WEIGHT·exc_mse − INH_WEIGHT·inh_mse
```

| term | typical size | why |
|---|---|---|
| recon | ~285 | squared error summed over 2×128×128 = 32,768 pixels |
| KL | ~50 | summed over 16 latents |
| exc_mse / inh_mse | 0 to ~1 | MSE on the standardized labels |

- The terms are just added together, so they don't have to share a range. What matters is their
  **relative size**, which decides what training prioritizes.
- Standardizing doesn't put exc/inh on recon's scale. It makes their scale **predictable** (0 to ~1 every run).
- The **weights** set how much each term counts. ×50 makes exc/inh about as large as KL, and recon still
  dominates on purpose. This is the same kind of choice as β in the β-VAE runs.
- Without the weights, exc/inh could change the loss by at most ~1 out of ~300, so training would ignore them.
- Example: with inh_mse at 0.70, fully defeating the adversary only gains 50 × (1 − 0.70) = 15 loss units
  against recon ≈ 285. That's one possible reason inhibition plateaued (try a larger `INH_WEIGHT` or more adversary steps).
