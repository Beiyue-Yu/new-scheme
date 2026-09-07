# MSTR
## Multi-Modal Spiking Tensor Regression Network for Audio-Visual Zero-Shot Learning
Zhe Yang, Wenrui Li, Jinxiu Hou, Guanghui Cheng. 
The code is based on [AVCA](https://github.com/ExplainableML/AVCA-GZSL) and tested on Ubuntu 20.04 with torch 2.0.1.

### Installing tensorly
Simply run, in your terminal:
```
pip install -U tensorly
```

### Inportant
The version of [spikingjelly](https://spikingjelly.readthedocs.io/zh_CN/latest/index.html) we used is 0.0.0.0.14.

Installing different versions can cause performance differences.
### Downloading features
The features and dataset structure could download and placed the same as [AVCA](https://github.com/ExplainableML/AVCA-GZSL).


## Evaluation
### Dowloading pre-trained models
[Here](https://drive.google.com/drive/folders/1A691fo9_DnKoTZJku4xTBfpgqw1Nn6Re?usp=sharing), you can download our trained MSTR models and baselines which are located in `pretrain_model.zip`
Put the content of `pretrain_model.zip` in the `runs/` folder.
### Test on three benchmark datasets
Here is an example for evaluating MSTR on Vggsound-GZSL using SeLaVi features.
``` 
python get_evaluation.py --load_path_stage_A runs/attention_ucf_vggsound_main --load_path_stage_B runs/attention_vggsound_all_main  --dataset_name VGGSound --MSTR 
```

## Results 
### GZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| APN                |    5.11                |    20.61        |   7.27           |
| VAEGAN             |    1.77                |    11.37        |   2.87           |
| SJE                |    2.15                |    26.50        |   5.57           |
| DEVISE             |    2.08                |    23.56        |   4.91           |
| CJME               |    6.17                |    12.48        |   5.12           |
| AVGZSLNET          |    5.83                |    18.05        |   6.44           |
| AVCA               |    6.31                |    27.15        |   12.13          |
| TCaF               |    7.33                |    31.72        |   10.71          |
| MSTR               |  **7.83**              |  **32.43**      | **13.21**        |


## Project Structure
```audioset_vggish_tensorflow_to_pytorch``` - Contains the code which is used to obtain the audio features using VGGish.

```c3d``` - Folder contains the code for the C3D network.

```selavi_feature_extraction``` - Contains the code used to extract the SeLaVi features.

```src``` - Contains the code used throughout the project for dataloaders/models/training/testing.

```cls_feature_extraction``` - Contains the code used to extract the C3D/VGGish features from all 3 datasets.

```avgzsl_benchmark_datasets``` - Contains the class splits and the video splits for each dataset for both features from SeLaVi and features from C3D/VGGish.

```splitting_scripts``` - Contains files from spltting our dataset into the required structure. 

```w2v_features``` - Contains the w2v embeddings for each dataset.
```run_scripts``` - Contains the scripts for training/evaluation for all models for each dataset.

## Reproduction and ablation modes

The current training entry point separates the MSTR control from the adaptive
time-step model:

- `--fusion_mode mstr_released`: released MSTR transformer and recurrence,
  with dynamic batch support and a pure-PyTorch TRL.
- `--fusion_mode mstr_paper`: corrected modality recurrence and the paper's
  8-head, 64-dimension CMF transformer.
- `--fusion_mode stft`: adaptive LIF/DTH/TSF model with GLP and LKC enabled.
- `--disable_glp` / `--disable_lkc`: isolate the contribution of GLP and LKC.
- `--cross_modal_residual`: opt-in, sample-adaptive complementary residual
  gating after Tucker fusion; it keeps the SNN/STFT pathway unchanged.
- `--semantic_contrastive_weight`: opt-in supervised multi-positive
  audio/video-to-text contrastive loss over the current Seen-class batch.
- `--global_prototype_contrastive_weight`: opt-in audio/video-to-text
  contrastive loss over the final task's complete Seen+Unseen class-name
  dictionary. It uses semantic prototypes only, never held-out AV examples.
- `--semantic_hard_negative_weight`: opt-in cosine-margin loss against each
  batch item's most semantically similar different Seen class. It addresses
  easy random-negative sampling without changing the SNN pathway.
- `--semantic_mixup_weight`: opt-in virtual-prototype loss that aligns mixed
  audio/video embeddings with a projection of mixed different-class Seen text.
- `--feature_mixup_weight`: opt-in raw feature/text manifold mixup through the
  full SNN-STFT encoder, using a paired different-class training negative as a
  local pseudo-unseen example.
- `--feature_debias_weight`: opt-in semantic/residual decomposition of the
  fused visual feature. It uses reconstruction, same-class semantic
  compactness, and an adversarial residual-to-text probe to suppress
  feature-level context variation without raw video frames.
- `--snn_membrane_readout_scale`: opt-in bounded residual from the final LIF
  membrane state. It preserves the hard-spike/DTH route while retaining
  sub-threshold temporal information that spike-only output discards.
- `--text_projection_norm layernorm`: replaces only `W_proj`'s text
  BatchNorm with per-prototype LayerNorm, preventing its running statistics
  from encoding a training-class-specific text distribution.

Each dataset has a two-stage baseline script. It uses batch size 256, the
paper dropout values, saves Stage B checkpoints, and runs matched evaluation:

```bash
bash run_scripts/UCF-GZSL/mstr_baseline_full.sh
```

The runner defaults to the isolated `MSTR-torch24` environment (PyTorch 2.4.1
with CUDA 12.1), restarts Python every five epochs, disables TensorBoard event
writing, and resumes the complete state from `last.pt`. It automatically retries
transient native-process failures up to five times without losing a completed epoch.
Override these settings with `PYTHON`, `EPOCH_CHUNK`, `SNN_T`, `SNN_TAU`, `STAGE_B_LR_SCALE`,
`DISABLE_TENSORBOARD`, or `MAX_STAGE_RETRIES` when needed. `STAGE_A_DIR` can
reuse a completed Stage A directory while retraining an independent Stage B.

Stage B retrains on the classes that were Unseen in Stage A. Its intermediate
validation loss is logged, but its GZSL HM is intentionally reported as `N/A`:
an HM on that split would treat training classes as Unseen and is invalid. The
runner selects the Stage B checkpoint matched to Stage A's selected epoch and
therefore trains Stage B only through that epoch; later Stage B checkpoints
cannot affect the matched evaluation.

The runner also holds a repository-wide GPU lock to prevent concurrent experiments.
It refuses the locally observed unsafe combination of Linux 6.17.0, the proprietary
NVIDIA 580.126.09 kernel module, and CUDA process teardown. Do not bypass this check
for normal experiments; change the host driver/module or kernel and reboot first.

Set `MODEL_MODE=mstr_released` to run the released-code-compatible control:

```bash
MODEL_MODE=mstr_released bash run_scripts/UCF-GZSL/mstr_baseline_full.sh
```

The STFT ablation scripts accept the GLP/LKC controls plus the optional
cross-modal residual configuration:

```bash
ABLATION=adaptive_only bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_glp  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=full          bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_cross_residual \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_semantic_contrastive \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_ceo_semantic_contrastive \
  bash run_scripts/UCF-GZSL/classwise_checkpoint_protocol.sh
ABLATION=adaptive_lkc_residual_snn_temporal_consistency \
  bash run_scripts/UCF-GZSL/classwise_checkpoint_protocol.sh
POSTHOC_FUSION=1 ABLATION=adaptive_lkc_residual \
  bash run_scripts/UCF-GZSL/classwise_checkpoint_protocol.sh

# Train-only semantic residuals are transferred to nearby class prototypes;
# temperature, transfer scale, and beta are frozen by Stage A class folds.
POSTHOC_RESIDUAL_PROTOTYPES=1 ABLATION=adaptive_lkc_residual \
  bash run_scripts/UCF-GZSL/classwise_checkpoint_protocol.sh
POSTHOC_CONDITIONAL_GENERATOR=1 ABLATION=adaptive_lkc_residual \
  bash run_scripts/UCF-GZSL/classwise_checkpoint_protocol.sh
POSTHOC_TEMPORAL_QUALITY_FUSION=1 ABLATION=adaptive_lkc_residual \
  bash run_scripts/UCF-GZSL/classwise_checkpoint_protocol.sh

The conditional-generator protocol is a lightweight, no-large-model
diagnostic. It fits a small semantic-to-audiovisual feature regressor from
training Seen samples only, selects Seen/Unseen prototype blending on
class-disjoint Stage A folds, and evaluates the matched Stage B checkpoint
once. Its UCF result was rejected after the independent test check; see
`research/conditional_feature_generator_result_20260805.md`.

Temporal-quality fusion is a second no-large-model diagnostic. It weights
audio/video distances with each STFT branch's semantic/SNN temporal agreement,
then freezes its gamma and beta through class-disjoint Stage A folds. It was
also rejected after independent Stage B testing; see
`research/temporal_quality_fusion_result_20260805.md`.
ABLATION=adaptive_lkc_residual_global_prototype_contrastive \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_semantic_hard_negative \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_semantic_mixup \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_feature_mixup \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_feature_debias \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_snn_activity_floor \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_membrane_readout \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_text_layernorm \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_stageb_init \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_stageb_seen_distill \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
ABLATION=adaptive_lkc_residual_stageb_group_balanced \
  bash run_scripts/UCF-GZSL/stft_ablation_full.sh
```

The CEO text-input ablation pre-optimizes only the final task's class-name
dictionary with semantic-ranking preservation and nearest-class separation.
It caches the resulting frozen text vectors under the dataset directory; no
audio/video test data is used. Run it with:

```bash
bash run_scripts/UCF-GZSL/ceo_text_full.sh
bash run_scripts/VGGSound-GZSL/ceo_text_full.sh
bash run_scripts/ActivityNet-GZSL/ceo_text_full.sh
```

Evaluation reads the dataset and model mode from Stage B's `args.pkl`; callers
only need the Stage A/Stage B paths and `--stage_b_selection matched`.

## Frozen OOD and fusion diagnostics

The following diagnostics keep the SNN and both MSTR checkpoints frozen. They
fit only the auxiliary three-layer Seen expert on concatenated SeLaVi
audio/video features, measure the EZ-AVOOD energy and PCA-residual signals,
and record per-sample fixed-sum fusion bias. GPU is the default; each command
uses the repository GPU lock and must run alone.

```bash
bash run_scripts/UCF-GZSL/ood_fusion_diagnostics.sh
bash run_scripts/VGGSound-GZSL/ood_fusion_diagnostics.sh
bash run_scripts/ActivityNet-GZSL/ood_fusion_diagnostics.sh
```

Each command writes `ood_fusion_diagnostics.json` and
`ood_fusion_test_samples.csv` into its Stage B run directory. The JSON reports
validation-selected OOD gamma/threshold, validation and test AUROC, frozen
two-expert routing accuracy, and whether fixed-sum fusion newly routes true
Unseen examples to Seen classes. The CSV provides the corresponding per-sample
predictions and Seen-minus-Unseen distance margins. Set `DEVICE=cpu` only when
GPU diagnostics are unavailable.

## Static class-conditioned evidence pilot

The pooled UCF cache contains one vector per modality and therefore cannot
support a real temporal claim. The independent evidence pilot treats those
vectors as static observations and scores every sample/class pair with shared
audio and video evidence heads. It compares the original class-name word2vec
against word2vec plus a versioned, label-only action signature. Neither path
uses SNN, TRL, Tucker fusion, class-specific parameters, or post-hoc beta.
Each episode builds pseudo-Seen support prototypes from query-disjoint samples;
a shared domain head learns a continuous Seen/Unseen prior jointly with class
matching instead of fitting a validation threshold.

The default protocol rotates three class-held-out folds entirely inside Stage
A's 30 training classes. Remaining Seen classes are split by complete UCF
video group, so no video group crosses training and evaluation. It does not
open the official Stage A validation or test split:

```bash
bash run_scripts/UCF-GZSL/class_conditioned_evidence_pilot.sh
```

The report records Seen/Unseen/HM/ZSL, per-fold Unseen coverage, dominant
prediction concentration, and true-class modality reliability. An explicit
`--protocol official-val` is available only after the internal design is
frozen; it still never opens Stage B or the test split.

The first internal pilot did not pass that gate, so official validation remains
unopened. See `research/class_conditioned_evidence_pilot_result_20260815.md`.

## Raw temporal LanguageBind pilot

The follow-up representation pilot no longer treats the pooled SeLaVi cache as
a temporal signal. It decodes raw UCF videos into three genuine temporal
thirds, aligns each third directly with class-label prompts using frozen
LanguageBind encoders, and caches the resulting observations with manifest and
checkpoint hashes. The audio path decodes the AVI's embedded waveform and uses
deterministic Mel windows; it does not inherit upstream random crop selection.

```bash
bash run_scripts/UCF-GZSL/languagebind_temporal_pilot.sh
```

On 369 Stage A training-only pilot videos, the frozen video temporal mean
reached 79.82% class-balanced accuracy and 71.69% worst-fold pseudo-unseen
accuracy. It passed the predeclared internal gate. Audio alone reached 18.33%,
but fixed equal AV fusion fell to 58.79%; global equal audio fusion was
therefore rejected rather than tuned on this pilot. Stage A validation, Stage
B, and test were still unopened when this internal result was frozen. See
`research/languagebind_temporal_av_pilot_result_20260815.md`.

After that gate passed, the official Stage A seen+unseen validation videos were
downloaded and validated. The frozen video-only temporal-mean evaluator has now
completed all 1820 videos with zero decode failures:

- Seen: 84.85%
- Unseen: 83.09%
- Harmonic mean: 83.96%
- ZSL: 84.57%

The predeclared official gate passed. The resumable runner uses bounded
50-video CUDA processes and can reproduce the result with:

```bash
bash run_scripts/UCF-GZSL/languagebind_official_val.sh
```

This command evaluates the predeclared 30% HM, 25% Unseen, 50% ZSL, and 75%
Unseen-coverage gate after all 1820 official validation videos are cached. The
machine-readable result is in
`runs/languagebind_ucf_stage1_val/report.json`. Stage B and test remain
unopened.

## Acknowledgement
We appreciate the code provided by [AVCA](https://github.com/ExplainableML/AVCA-GZSL), which is very helpful to our research.
