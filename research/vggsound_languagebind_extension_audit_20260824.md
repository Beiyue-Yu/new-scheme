# VGGSound LanguageBind extension audit (2026-08-24)

## Scope

The goal was to extend the validated UCF route to VGGSound:

`LanguageBind_Video_FT` video embeddings + LanguageBind text prototypes + the
repository's existing 512-D audio observations + MSTR training.

## What is available locally

- VGGSound main split contains 93,752 unique sample filenames across 309
  classes.
- Existing processed `main_features` files cover 70,351 training samples,
  10,919 validation samples, 81,270 train+validation samples, and 12,482
  test samples.
- Existing audio and video observations in `main_features` are both 512-D;
  existing text is 300-D.
- LanguageBind Video_FT and Audio_FT checkpoints are present locally.

## Work performed

`extract_languagebind_vggsound.py` was added. It:

- reads all six main-split manifests;
- validates filename, class-id and label consistency;
- downloads temporal clips by YouTube ID and start offset when local videos
  are absent;
- extracts three temporal segments with the frozen LanguageBind video encoder;
- stores 768-D segment embeddings, 768-D prompt-ensemble text prototypes,
  provenance, failures and resumable coverage metadata;
- never falls back to the old 512-D video cache.

The downstream `build_languagebind_mstr_features.py` can then join the cache to
the four processed pickle files by stable filename and retain the original
audio observations.

The two-stage runner now infers dimensions from the selected feature route:
`languagebind_mstr` defaults to video 768-D and text 768-D, while legacy
`main_features`/`cls_features` retain video 512-D and text 300-D. Explicit
`INPUT_SIZE_VIDEO`, `INPUT_SIZE_AUDIO`, and `TEXT_EMBEDDING_SIZE` values still
override these defaults. Both the shell runner and `main.py` now perform a
pre-training cache dimension check, so a stale 512-D model cannot reach the
first training batch against a 768-D cache.

## Blocking state

The workspace has no VGGSound raw videos (`local_videos_present=0/93752`). The
manifest and cache audit passed, and `yt-dlp` is installed in the MSTR
environment, but a live request to YouTube fails with:

`URLError: [Errno 101] Network is unreachable`

A one-sample LanguageBind pilot successfully loaded the local model and
encoded all 309 text prototypes, then could not obtain the first video. No
VGGSound LanguageBind video cache was written and no VGGSound LanguageBind
training was started. This is intentional: zero-filled or reused 512-D video
features would invalidate the experiment.

## Downloader CSV audit

The supplied `raw_datasets/VGGSound/download/vggsound.csv` contains 199,176
rows with YouTube ID, start second, label, source split and row number. Its
IDs can be converted directly to downloader URLs of the form
`https://youtube.com/watch?v=<youtube_id>`.

The filtered task manifest
`raw_datasets/VGGSound/download/vggsound_main_split.csv` matches 93,732 of the
93,752 current main-split samples. Twenty samples are absent from this CSV,
which indicates a metadata-version difference rather than a filename join
failure. Their exact IDs and offsets are recorded in
`vggsound_main_split.csv.report.json`. A strict complete experiment must either
obtain those 20 rows from a matching VGGSound metadata revision or explicitly
exclude them and report the reduced coverage.

The downloader writes clips as `train/test/<normalized-label>/v<ID>_<start>_<end>_out.mkv`.
The extractor now recognizes that layout in addition to flat MP4 and
`label/filename` layouts.

## Artifacts

- Extractor: `extract_languagebind_vggsound.py`
- Coverage report: `runs/vggsound_languagebind_feature_audit_20260824/coverage.json`
- Existing source cache: `avgzsl_benchmark_datasets/VGGSound/_features_processed/main_features/`

To resume, provide the complete VGGSound source videos locally under a
directory named by `--video_root`, or run the extractor on a machine with
network access and yt-dlp. Only after coverage reaches 93,752/93,752 should the
LanguageBind cache be converted and the full MSTR protocol launched.
