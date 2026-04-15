# Baguan-solar Paper and Repo Audit

## Scope

This note aligns the paper `Integrating Weather Foundation Model and Satellite to Enable Fine-Grained Solar Irradiance Forecasting` with the cloned repo `https://github.com/DAMO-DI-ML/Baguan-solar`, then lists implementation details, reproducibility blockers, and the next practical steps.

Paper metadata checked during this audit:

- Title: `Integrating Weather Foundation Model and Satellite to Enable Fine-Grained Solar Irradiance Forecasting`
- arXiv: `2603.14845v2`
- Latest arXiv revision seen in the PDF: `2026-03-17`

## One-paragraph understanding

Baguan-solar is a two-stage multimodal solar irradiance forecasting system. Stage 1 fuses `6 h` historical Himawari satellite observations with `30 h` large-scale weather fields to predict `24 h` future cloud-related intermediates: total cloud cover and future satellite imagery. Stage 2 then maps these cloud-aware intermediates, plus clear-sky GHI and radiation-relevant meteorological variables, to `24 h` high-resolution GHI. The central design logic is:

- satellite imagery preserves fine-scale cloud morphology;
- Baguan or ERA5 supplies large-scale dynamical constraints;
- cloud prediction is easier to model day/night continuously than raw GHI;
- irradiance inference becomes easier after cloud evolution is decoupled.

## What the paper claims

### Data

- Satellite: Himawari-8/9 AHI, 4 bands: `B03`, `B07`, `B10`, `B14`
- Target: CLDAS `SSRD -> GHI`, plus CLDAS `TCDC`
- Training-time meteorology: ERA5
- Inference-time meteorology: Baguan forecasts
- Spatial target grid: `512 x 512`, about `0.05°`
- Temporal setup: `6 h` history, `24 h` forecast horizon

### Model

- Stage 1:
  - inputs: satellite history + `39` meteorological channels over `30` time steps
  - dual encoders for morphology and environment
  - cross-attention from satellite tokens to meteorological tokens
  - decoder outputs `24 x (4 satellite bands + 1 TCDC)`
- Stage 2:
  - inputs: predicted cloud, predicted satellite, clear-sky GHI, and `11` radiation-relevant meteorological channels
  - outputs: `24 h` GHI
- Backbone family: Swin Transformer

### Results reported in the paper

- Best operational variant: `Baguan-solar (oper.)`
- Table 2 average RMSE:
  - `Two-stage Swin`: `49.89`
  - `Baguan-solar (oper.)`: `41.87`
- Claimed relative improvement over strongest baseline: about `16.08%`
- Operational deployment claim:
  - online since `July 2025`
  - Baguan runs `4` times per day: `UTC 00, 06, 12, 18`
  - Baguan-solar runs hourly on top of latest Baguan + satellite data

## What the repo actually contains

Tracked files are minimal and focused on the main training path:

- `train_BaguanSolar.py`
- `test_BaguanSolar.py`
- `model/BaguanSolar.py`
- `datasets/two_stage_dataset.py`
- `datasets/data_preprocessing/*.py`
- `utils/*.py`

What is not present in tracked files:

- baseline training code for `SolarSeer`, `EC IFS`, `Two-stage Unet`, or `Two-stage Swin`
- ablation scripts for `w/o Baguan`, `w/o satellite`, `Only S1`, `w/o TCDC`
- Integrated Gradients analysis code
- deployment pipeline code
- pyranometer evaluation code

Conclusion: this repo is the core model repo, not a full paper reproduction package.

## Paper-to-code mapping

### Stage 1 mapping

From `model/BaguanSolar.py`:

- `self.sate_encoder`: satellite encoder
- `self.env_encoder`: environment encoder
- `self.cross_sat`: cross-attention fusion
- `self.decoder_1`: joint decoder for cloud + satellite future

Implementation details match the paper well:

- satellite input channels: `6 * 4 = 24`
- environment Stage 1 input channels: `30 * 39 = 1170`
- decoder output channels: `24 * (4 + 1) = 120`

### Stage 2 mapping

From `model/BaguanSolar.py`:

- future meteorological slice indices:
  - `7..13`: `Q` at 7 pressure levels
  - `37, 38, 39, 40`: `TCW`, `TCWV`, `SSRD`, `FDIR`
- this gives `11` channels, which matches the paper's Stage 2 meteorological subset

### Data mapping

From preprocessing and dataset code:

- merged `6` channels:
  - CLDAS: `SSRA`, `TCDC`
  - Himawari: `albedo_03`, `tbb_07`, `tbb_10`, `tbb_14`
- clear-sky GHI:
  - computed by a custom vectorized implementation in `datasets/cal_clear_ghi.py`
- training:
  - uses ERA5
- testing / operational emulation:
  - uses `baguan_dir`

### Channel semantics from preprocessing order

`datasets/data_preprocessing/era5_preprocess.py` implies the `41` meteorological channels are ordered as:

1. `Z` at `925, 850, 700, 600, 500, 250, 50`
2. `Q` at the same `7` levels
3. `T` at the same `7` levels
4. `U` at the same `7` levels
5. `V` at the same `7` levels
6. `LCC`
7. `TCC`
8. `TCW`
9. `TCWV`
10. `SSRD`
11. `FDIR`

This explains:

- Stage 1 uses the first `39` channels and excludes the last two radiation channels
- Stage 2 explicitly uses `Q + TCW + TCWV + SSRD + FDIR`

## Important mismatches and reproduction risks

### 1. Training loop currently stops after one batch per epoch

In `train_BaguanSolar.py`, the inner training loop contains an unconditional `break`.

Impact:

- each epoch uses only the first batch;
- logged epoch losses are divided by the full number of steps anyway;
- a direct run of this repo will not reproduce the paper training behavior.

### 2. Loss weights do not match the paper

The paper states:

- `lambda_sat = 1`
- `lambda_TCDC = 0.5`
- `lambda_ghi = 1`

The training code uses:

- `loss = aux_loss_weight * loss_cloud + loss_satellite + loss_ghi`
- default `aux_loss_weight = 0.1`

Impact:

- cloud supervision is much weaker than the paper description unless overridden manually.

### 3. Stage 2 code uses an extra lead-time channel not stated in the paper

The appendix says Stage 2 input channels are:

- predicted cloud: `1`
- predicted satellite: `4`
- clear-sky GHI: `1`
- meteorology: `11`
- total: `17`

But the code sets:

- `in_chans = 3 + 4 + 11 = 18`

because it concatenates:

- cloud
- clear-sky GHI
- `lead_time`
- predicted satellite
- 11 meteorological channels

Impact:

- the implementation is not identical to the appendix description;
- this is likely an undocumented enhancement rather than a bug.

### 4. Dataset filtering keeps only `UTC 00` and `UTC 12` initializations

`datasets/two_stage_dataset.py` filters sequence anchors to:

- `00`
- `12`

But the paper states lead-time curves are shown for:

- `UTC 00`
- `UTC 06`
- `UTC 12`
- `UTC 18`

Impact:

- current repo behavior does not cover all initialization times reported in the paper figures.

### 5. Continuity check is incomplete

The dataset builder does:

- if sequence is continuous, then validate ERA5 coverage;
- but if sequence is not continuous, it still appends the sequence.

Impact:

- broken or gapped temporal sequences can leak into the dataset if files are missing.

### 6. Repo does not contain full paper evaluation tooling

The paper reports:

- multiple baselines
- ablations
- qualitative case comparisons
- modality attribution by Integrated Gradients
- deployment verification with `246` pyranometer sites

The public repo does not contain the corresponding reproduction code.

Impact:

- the repo is enough to understand and probably train the main model;
- it is not enough to reproduce the full paper tables and figures end to end.

## Practical interpretation of the method

The paper's real contribution is not just "Baguan + satellite". It is the combination of three design choices:

1. `Decoupling`
   Predict cloud-related intermediates first, then infer irradiance.

2. `Asymmetric modality usage over horizon`
   Satellite dominates short lead times; Baguan or ERA5 dominates longer lead times.

3. `Physics-informed inference`
   Clear-sky GHI acts as a physically meaningful prior and day/night mask.

The ablation table and the modality analysis tell a consistent story:

- removing meteorological forcing hurts long lead times most;
- removing satellite hurts short lead times most;
- removing the two-stage design hurts both stability and physical interpretability.

This is the most important conceptual takeaway from the paper.

## What I currently trust vs. what I do not

### High confidence

- the overall architecture in the repo matches the paper's main idea;
- preprocessing logic is consistent with the paper's dataset description;
- channel counts and variable groups mostly align with the appendix;
- the repo is likely the official core implementation, not an unrelated re-implementation.

### Lower confidence / needs care before reproduction

- exact paper training recipe
- exact paper evaluation subset
- exact loss weighting used in the reported results
- full baseline comparability
- whether the published repo was cleaned after internal experiments

## Recommended next work cycles

### Cycle 1: make the public repo minimally reproducible

- remove the accidental one-batch `break`
- align loss weights with the paper or expose them clearly
- fix the continuity filter
- document the exact expected Baguan file naming contract
- add a reproducibility README with data shape and channel order

### Cycle 2: run sanity checks

- verify one full train epoch iterates over all batches
- verify Stage 1 and Stage 2 tensor shapes on a small synthetic sample
- verify `00/12` vs `00/06/12/18` initialization logic
- verify inference with Baguan inputs matches the paper's operational setting

### Cycle 3: rebuild missing evaluation pieces

- add baseline runners if available
- add ablation switches
- add figure generation scripts
- add site-level verification scripts if station data exist

## Bottom line

This repo is valuable for understanding the Baguan-solar model itself, especially the two-stage design and the meteorology-satellite fusion logic. It is not yet a clean, one-command reproduction package for the entire paper. If used as-is, it is best treated as:

- an official core implementation,
- a strong starting point for reproduction,
- but not a final benchmark-grade release.
