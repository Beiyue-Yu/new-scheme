# LanguageBind-to-MSTR integration result (2026-08-18)

## Question

Can the passing frozen LanguageBind video representation be used as the MSTR
video and text input while retaining MSTR's audio branch and frozen UCF GZSL
selection protocol?

## Route

- Audio: original 512-D repository feature.
- Video: normalized mean of three frozen LanguageBind temporal segments,
  768-D.
- Text: frozen LanguageBind prompt ensemble, 768-D.
- MSTR: `adaptive_lkc_residual`, with its text input/reconstruction dimension
  generalized from 300 to 768. The default remains 300 for old checkpoints.
- Seed: 1 for the complete frozen protocol.
- Stage A: 50 epochs and all 50 checkpoints.
- Selection: three class-disjoint validation folds, followed by frozen
  calibration.
- Stage B: matched selected epoch only.
- Test: loaded once after Stage A selection and Stage B training.

All required raw videos were validated and encoded with zero failures:

| Split | Videos |
| --- | ---: |
| Stage A train | 3174 |
| Stage A validation | 1820 |
| Stage 2 test | 1822 |

The generated feature manifest reports zero missing video samples and zero
missing active class texts for every processed split.

## End-to-end result

Stage A selected epoch 4. The accepted Both calibration used beta 0.4 and
classwise strength 0.75. Its held-out-fold HM was unstable: the three fold HMs
were 0.79%, 33.17%, and 0.00%.

| MSTR output | Seen | Unseen | HM |
| --- | ---: | ---: | ---: |
| Audio | 43.97% | 14.11% | 21.36% |
| Video | 54.15% | 6.67% | 11.88% |
| Both | 54.15% | 6.57% | 11.72% |

The matched repository reference is Both Seen 57.91%, Unseen 18.84%, and HM
28.43%. Direct feature replacement therefore changes HM by **-16.71 points**.
The main failure is Unseen transfer rather than Seen recognition.

## Frozen direct diagnostic control (not a formal comparison)

The unchanged LanguageBind temporal-mean classifier was also evaluated on the
same Stage 2 manifests with the prompts and aggregation frozen before the test
run. This was **not a complete training experiment**: no MSTR Stage A/Stage B
training, checkpoint selection, or matched trained baseline was performed for
this route. It is retained only as a representation diagnostic and is excluded
from all formal baseline, improvement, and paper-table comparisons.

| Frozen LanguageBind | Seen | Unseen | HM | ZSL |
| --- | ---: | ---: | ---: | ---: |
| Temporal mean | 81.51% | 98.40% | 89.16% | 98.76% |

This diagnostic suggests that the source representation is highly
discriminative, but it cannot establish a trained-system comparison. The
trained MSTR result still stands on its own under the complete frozen protocol;
the direct diagnostic must not be used to claim an improvement or an upper
bound for the final method.

`LanguageBind_Video_FT` is the upstream fully fine-tuned checkpoint trained on
large web video-text data. The local provenance does not show task-specific
UCF101 fitting, but public-video overlap cannot be excluded. This diagnostic is
therefore not a paper result; provenance and contamination checks remain open.

## Data-alignment correction (2026-08-22)

The original feature builder assigned LanguageBind text rows by their position
in the source dictionary. UCF's `text.target` list has a reordered tail (class
20 is not at row 20), while the dataset loader indexes text rows directly by
class ID. This silently gave some classes the wrong prototype in the run
above. The processed features were rebuilt with explicit class-ID assignment,
and the old 11.72% trained result is therefore retained only as a historical
failure symptom, not as a valid method comparison. The frozen direct diagnostic
was always excluded from formal comparisons and remains so.

The corrected route is documented in
`research/languagebind_anchor_residual_result_20260822.md`.

## Decision

Reject direct LanguageBind feature replacement inside the current MSTR
objective. Do not tune this route further. The next candidate should preserve
the frozen LanguageBind text-similarity logits as a direct branch and restrict
MSTR to a validation-gated residual correction. A valid correction must fall
back exactly to the frozen branch when it does not improve class-held-out
validation folds.

## Artifacts

- Processed feature manifest:
  `avgzsl_benchmark_datasets/UCF/_features_processed/languagebind_mstr/manifest.json`
- Stage A selection:
  `runs/ucf_stft_adaptive_lkc_residual_val_languagebind_mstr_full_20260818/classwise_checkpoint_selection.json`
- End-to-end frozen test:
  `runs/ucf_stft_adaptive_lkc_residual_all_languagebind_mstr_full_20260818/classwise_calibration_test.json`
- Frozen direct diagnostic (excluded from formal comparisons):
  `runs/languagebind_ucf_stage2_test_mstr/direct_report.json`
