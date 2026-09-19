import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHT_SHA256 = (
    "9530818dfd77c754d84ad7425648d0fb4a3cef662b097a3f300dffce80efde40"
)


class ReleaseLayoutTest(unittest.TestCase):
    def test_pretrained_weight_checksum(self):
        model_path = ROOT / "weights" / "pcard_pretrained.pth"
        self.assertTrue(model_path.is_file())
        self.assertEqual(
            hashlib.sha256(model_path.read_bytes()).hexdigest(),
            EXPECTED_WEIGHT_SHA256,
        )

    def test_no_patient_data_file_types(self):
        forbidden = {".csv", ".h5", ".hdf5", ".zip"}
        found = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden
        ]
        self.assertEqual(found, [])

    def test_no_wandb_key_literal(self):
        # W&B API keys are 40-character hexadecimal strings. Ignore binary
        # weights and Git internals when checking release text.
        candidate = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", re.I)
        matches = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix == ".pth":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if candidate.search(text):
                matches.append(str(path.relative_to(ROOT)))
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
