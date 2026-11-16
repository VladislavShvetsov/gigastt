import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prepare', SCRIPTS / 'prepare_fp32_model.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ModelPackTests(unittest.TestCase):
    def test_hash_reaches_end_of_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights'
            data = b'model' * 300000
            path.write_bytes(data)
            result = subprocess.run(
                [sys.executable, '-c',
                 'from prepare_fp32_model import sha; from pathlib import Path; '
                 'import sys; print(sha(Path(sys.argv[1])))', str(path)],
                cwd=SCRIPTS, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), hashlib.sha256(data).hexdigest())

    def test_prepare_verify_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename in prepare.FILES.values():
                (root / filename).write_bytes(b'model-fixture')
            result = self.prepare(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = tomllib.loads((root / 'manifest.toml').read_text(encoding='utf8'))
            self.assertEqual(manifest['precision'], 'fp32')
            result = self.verify(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            (root / prepare.FILES['encoder']).write_bytes(b'corrupted')
            self.assertNotEqual(self.verify(root).returncode, 0)

    def test_reject_int8_in_fp32_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename in prepare.FILES.values():
                (root / filename).write_bytes(b'model-fixture')
            (root / 'v3_e2e_rnnt_encoder_int8.onnx').write_bytes(b'int8')
            self.assertNotEqual(self.prepare(root).returncode, 0)

    def test_reject_unpinned_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(self.prepare(Path(tmp), 'main').returncode, 0)

    @staticmethod
    def prepare(root, revision='a' * 40):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / 'prepare_fp32_model.py'),
             '--no-download', '--output', str(root), '--revision', revision],
            capture_output=True, text=True, timeout=10,
        )

    @staticmethod
    def verify(root):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / 'verify_fp32_model.py'), str(root)],
            capture_output=True, text=True, timeout=10,
        )


if __name__ == '__main__':
    unittest.main()
