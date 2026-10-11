"""Prevent research-history filenames and imports from entering the release."""

from __future__ import annotations

import re
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSIONED_NAME = re.compile(r"(?:^|_)[vV]\d+(?:_|\.|$)")
HISTORICAL_IMPORT = re.compile(r"\b(?:diagnose|train|eval)_[vV]\d+")
HISTORICAL_VERSION_TEXT = re.compile(r"\bV\d+(?:\.\d+)?[a-z]?\b")
NON_RELEASE_NAME = re.compile(
    r"(?:^|_)(?:debug|diagnose|diagnostic|pilot|smoke|rehearsal|temporary|tmp)(?:_|\.|$)",
    re.IGNORECASE,
)


class SubmissionHygieneTest(unittest.TestCase):
    def test_python_filenames_are_descriptive(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*.py")
            if VERSIONED_NAME.search(path.name)
        ]
        self.assertEqual(offenders, [])

    def test_python_filenames_are_release_facing(self) -> None:
        offenders = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*.py")
            if NON_RELEASE_NAME.search(path.name)
        ]
        self.assertEqual(offenders, [])

    def test_no_historical_imports(self) -> None:
        offenders = []
        for path in ROOT.rglob("*.py"):
            if HISTORICAL_IMPORT.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_matlab_generators_have_no_historical_version_labels(self) -> None:
        offenders = []
        for path in (ROOT / "simulation").glob("*.m"):
            if HISTORICAL_VERSION_TEXT.search(
                path.read_text(encoding="utf-8-sig")
            ):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_simulation_sources_are_release_facing(self) -> None:
        self.assertEqual(list((ROOT / "data").glob("*.m")), [])
        expected = {
            "generate_source_count_training.m",
            "generate_session_calibration_training.m",
            "generate_external_session_test.m",
            "generate_ood_tests.m",
            "generate_close_source_tests.m",
            "generate_calibration_drift_tests.m",
            "generate_structured_waveform.m",
        }
        actual = {path.name for path in (ROOT / "simulation").glob("*.m")}
        self.assertEqual(actual, expected)
        for path in (ROOT / "simulation").glob("*.m"):
            self.assertFalse(VERSIONED_NAME.search(path.name), path.name)

    def test_simulation_outputs_match_protocol(self) -> None:
        protocol = json.loads(
            (ROOT / "configs" / "final_protocol.json").read_text(encoding="utf-8")
        )
        sources = "\n".join(
            path.read_text(encoding="utf-8-sig")
            for path in (ROOT / "simulation").glob("*.m")
        )
        for filename in protocol["datasets"].values():
            if filename.startswith("drift_"):
                self.assertIn("output_name = ['drift_', drift_component, '.mat']", sources)
            else:
                self.assertIn(filename, sources)

    def test_removed_refiner_is_absent(self) -> None:
        offenders = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*iq_refin*")]
        self.assertEqual(offenders, [])

    def test_no_machine_specific_paths(self) -> None:
        windows_path = re.compile(r"[A-Za-z]:\\\\")
        offenders = []
        for path in ROOT.rglob("*.py"):
            if windows_path.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
