# VGGSound LanguageBind MSTR baseline (2026-09-07)

This document freezes the current comparison baseline before any variable-time
sequence changes are made.

## Definition

- Dataset: VGGSound `main_split`, 309 classes.
- Audio input: original SeLaVi 512-D features.
- Video input: frozen LanguageBind Video_FT, 768-D.
- Text input: frozen LanguageBind prompt-ensemble prototypes, 768-D.
- Model: the repository STFT MSTR route, `adaptive_lkc_residual`.
- Internal simulation steps: fixed `SNN_T=10`.
- Training: Stage A validation training, matched Stage B retraining, then
  repository evaluation.
- Data directory: `avgzsl_benchmark_datasets/VGGSound/_features_processed/`
  `languagebind_mstr_partial_20260907`.

The LanguageBind cache contains three temporal video segments per sample, but
the frozen baseline converts them to their normalized mean before MSTR. The
baseline therefore does not claim variable-length temporal modeling.

## Data coverage

The source cache contains 89,003 of 93,752 manifest videos (94.93%). The
corrupt `vggsound_08.tar.gz` shard accounts for all missing records:

| split | samples | missing video vectors |
| --- | ---: | ---: |
| Stage 1 train | 70,351 | 3,530 |
| Stage 1 validation | 10,919 | 572 |
| Stage 2 train | 81,270 | 4,102 |
| Stage 2 test | 12,482 | 647 |

Missing vectors are zero placeholders in the partial processed directory. The
experiment must be reported as a partial-data baseline, not as an official
complete VGGSound result.

## Reproduction

Build the processed files from the frozen cache:

```bash
python build_languagebind_mstr_features.py \
  --processed_source avgzsl_benchmark_datasets/VGGSound/_features_processed/main_features \
  --processed_destination avgzsl_benchmark_datasets/VGGSound/_features_processed/languagebind_mstr_partial_20260907 \
  --cache runs/languagebind_features/VGGSound/vggsound_main_segments.npz \
  --allow_missing
```

Run the two-stage baseline in the foreground:

```bash
bash run_scripts/VGGSound-GZSL/languagebind_mstr_partial_baseline.sh
```

The default run directory is
`runs/vggsound_languagebind_partial_stft_adaptive_lkc_residual_val_20260907_1427`
for Stage A and the matching `_all_...` directory for Stage B. Training is
chunked in five-epoch processes and resumes from `last.pt`.

## Frozen diagnostic reference

Before MSTR training, frozen LanguageBind video-text matching on the available
Stage 2 test records produced Seen 48.99%, Unseen 37.02%, HM 42.17%, and ZSL
54.42% at 94.82% test coverage. This is a diagnostic only and is not the MSTR
baseline score.

## Local artifact checksums

Large artifacts are intentionally not committed to Git. Their current local
locations and checksums are:

- cache: `runs/languagebind_features/VGGSound/vggsound_main_segments.npz`
  (`e7de6c22c068df759e9b50dbdbfa45b8cd4131eb02eeb6fa5c6d62c234578123`)
- processed manifest:
  `avgzsl_benchmark_datasets/VGGSound/_features_processed/`
  `languagebind_mstr_partial_20260907/manifest.json`
  (`67e6c6be5c13581eac701a2fcfe3b598cb22819740730546c6dd6a4caf89c1d7`)
