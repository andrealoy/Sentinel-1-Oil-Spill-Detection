# Sentinel-1 Oil Spill Detection (SAR)

Deep learning project for detecting marine oil spills from Sentinel-1 SAR image patches.

## Why this project

Oil spill detection is safety-critical and environmentally sensitive. Synthetic Aperture Radar (SAR) is a good fit because it works day/night and through clouds. In SAR imagery, oil films often dampen short surface waves, making affected areas smoother and darker than surrounding water.

This repository explores that hypothesis end-to-end:

1. **EDA + outlier analysis** to validate physical intuition.
2. **Frequency-domain analysis (FFT)** to quantify textural differences.
3. **Model training** from simple baselines to CNN/ResNet.

---

## Dataset and challenge

- Binary classification:
	- `0` = non-oil
	- `1` = oil
- Input format: grayscale SAR patches (`400x400`), with class imbalance and potential acquisition artifacts.
- Main practical issue: avoiding models that overfit spurious cues (e.g., extreme saturation artifacts) instead of true oil signatures.

---

## Why we split by region (to avoid model “cheating”)

A random image-level split is risky for this dataset because many patches come from nearby coordinates or the same acquisition scene.

If patches from the same region appear in both train and test:

- the model can memorize local background patterns (coastline geometry, acquisition artifacts, texture fingerprints),
- evaluation becomes artificially optimistic,
- and reported performance may not transfer to truly unseen areas.

To reduce this leakage, we use a **spatial/region-aware split**:

- patches are grouped by region/scene identifier extracted from filenames,
- entire groups are assigned to `train`, `val`, or `test` (not split across them),
- then models are evaluated on regions not seen during training.

In short: this split strategy tests **generalization to new geographic contexts**, not just memorization of recurring spatial signatures.

Related utilities in `utils.py`:

- `extract_scene_id(...)`
- `create_spatial_train_test_val_split(...)`
- `create_spatial_split_full(...)`
- `count_images_by_region(...)`

---

## What the exploratory analysis found (`eda_outliers_fft.ipynb`)

The EDA highlights consistent class-level differences:

- **Oil patches are darker on average** than non-oil patches.
- **Oil patches have lower variance / smoother texture**.
- **Pixels saturated at intensity 255 are much more frequent in class 0**, with extreme outliers in non-oil.
- **FFT analysis confirms texture differences**:
	- both classes are dominated by low frequencies,
	- but non-oil tends to retain more high-frequency energy.

Quantitatively, the notebook reports a slightly higher mean high-frequency ratio for non-oil (around `0.56`) than oil (around `0.53`), consistent with rougher sea texture outside oil films.

### EDA takeaway

The signal is real but subtle. Global descriptors help, yet they are not sufficient alone; spatial structure matters, which motivates convolutional models.

---

## Modeling pipeline (`model_training.ipynb`)

The training notebook builds a progression:

1. **Feature baseline** (global stats + logistic-style sanity checks).
2. **MLP baseline**.
3. **Simple CNN**.
4. **ResNet18 transfer-learning style setup**.

It also addresses common training concerns:

- strict train/val/test organization,
- duplicate and corrupted-file checks,
- optional class weighting / balancing,
- F1-driven model selection (`val_f1` used as target metric in training loops).

### Reported results (from notebook runs)

On a held-out test split (support shown as `218` in the notebook reports), the progression improves substantially:

- **MLP**: accuracy around `0.88`
- **Simple CNN**: accuracy around `0.95`
- **ResNet18**: accuracy around `0.97`

Validation metrics in the logs show the same trend, with higher best `val_f1` for CNN/ResNet compared with MLP.

---

## Repository structure

- `eda_outliers_fft.ipynb` — exploratory analysis, outlier profiling, FFT-based frequency analysis.
- `model_training.ipynb` — training/evaluation of MLP, CNN, and ResNet variants.
- `utils.py` — reusable utilities for cleaning, splitting, balancing, diagnostics, evaluation, and visualization.
- `data_fetch.py` — dataset retrieval/preparation helper.
- `requirements.txt` / `pyproject.toml` — environment dependencies.
- `model/` — saved model checkpoints.
- `data/organized_data/` — train/val/test folderized data.

---

## `utils.py` function reference

The following utilities support the full workflow.

### Data quality and cleanup

- `clean_low_variance_dataset(root_dir, threshold=2.0)`
	- Removes low-information images by variance threshold.
- `find_and_remove_black_images(data_dir, threshold=1.0, delete=False)`
	- Detects nearly black images; can optionally delete them.
- `fix_images(data_dir)`
	- Verifies and normalizes image files/extensions.
- `check_duplicates(*directories)`
	- Detects duplicate image content across directories.
- `remove_duplicates(*directories)`
	- Removes duplicated files found by content checks.

### Class distribution and quick diagnostics

- `show_class_distribution(data_dir)`
	- Prints per-class image counts.
- `plot_class_distribution(data_dir, title="Class Distribution")`
	- Visualizes class counts.
- `show_img_size_and_channels(img_path)`
	- Displays shape/channels metadata for one image.
- `display_random_samples(root_path, n_samples=5, max_char=15)`
	- Shows random examples by class.

### SAR preprocessing and denoising

- `preprocess_sar(image_np)`
	- SAR-specific intensity preparation (normalization/cleanup steps).
- `denoise_sar(image_np)`
	- Denoising helper for SAR-like textures.

### Splitting strategies (including leakage-aware spatial splits)

- `create_train_test_val_split(...)`
	- Standard random split into train/val/test folders.
- `create_spatial_train_test_val_split(...)`
	- Spatially-aware split to reduce geographic leakage.
- `create_spatial_split_full(...)`
	- Extended spatial split routine used in notebook experiments.
- `count_images_by_region(data_dir)`
	- Region-level counting for spatial diagnostics.
- `extract_scene_id(filename_stem)`
	- Parses scene/region identifier from filenames.

### Training support and balancing

- `set_seed(seed=42)`
	- Reproducibility helper.
- `get_mean_and_std(loader)`
	- Computes dataset normalization stats.
- `get_class_weights(dataset)`
	- Builds inverse-frequency class weights.
- `create_balanced_sampler(dataset)`
	- Returns a weighted sampler for imbalance mitigation.
- `verify_batch(loader, std, mean)`
	- Visual sanity check of transformed mini-batches.

### Evaluation and explainability

- `evaluate_test_set(model, test_loader, device)`
	- Produces test metrics/report artifacts.
- `visualize_random_gradcam(model, target_layer, dataset, device, num_images=5)`
	- Grad-CAM overlays for qualitative interpretability.

### History / learning curves

- `plot_history(hist)`
	- Generic training curve plotting.
- `plot_pytorch_history(history)`
	- PyTorch-specific history plotting helper.

### Label/path consistency checks

- `get_files_by_class(data_dir)`
	- Lists files grouped by class directory.
- `sanity_check_labels(dataset, original_data_dir, max_checks=None)`
	- Cross-checks dataset labels against source folder structure.

---

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open notebooks in this order:

1. `eda_outliers_fft.ipynb`
2. `model_training.ipynb`

---

## Notes

- Reported metrics depend on split strategy, random seed, and checkpoint selection.
- For operational oil-spill screening, prioritize recall-sensitive monitoring and threshold tuning, not only global accuracy.
