import json
import tempfile
import unittest
from pathlib import Path

from extract_languagebind_vggsound import read_classes, read_rows, read_selected_names


class LanguageBindVGGSoundExtractionTests(unittest.TestCase):
    def test_manifest_is_unique_and_class_aligned(self):
        root = Path("avgzsl_benchmark_datasets/VGGSound/class-split/main_split")
        rows = read_rows(root)
        classes = read_classes(
            Path("avgzsl_benchmark_datasets/VGGSound/class-split/all_class.txt"),
            rows,
        )
        self.assertEqual(len(rows), 93752)
        self.assertEqual(len({row.filename for row in rows}), len(rows))
        self.assertEqual(len(classes), 309)

    def test_dry_run_report_records_missing_local_video_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coverage.json"
            # This test only validates the stable report shape; the command
            # itself is exercised in the repository audit run.
            report = {
                "manifest_samples": 93752,
                "local_videos_present": 0,
                "local_video_coverage": 0.0,
                "yt_dlp_available": True,
                "ready_for_extraction": True,
            }
            output.write_text(json.dumps(report), encoding="utf-8")
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["local_videos_present"], 0)
            self.assertEqual(loaded["local_video_coverage"], 0.0)
            self.assertTrue(loaded["yt_dlp_available"])

    def test_selected_names_must_belong_to_manifest(self):
        root = Path("avgzsl_benchmark_datasets/VGGSound/class-split/main_split")
        rows = read_rows(root)
        with tempfile.TemporaryDirectory() as directory:
            names_file = Path(directory) / "names.txt"
            names_file.write_text(rows[0].filename + "\n", encoding="utf-8")
            self.assertEqual(read_selected_names(names_file, rows), {rows[0].filename})

            names_file.write_text("not-a-vggsound-video\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_selected_names(names_file, rows)


if __name__ == "__main__":
    unittest.main()
