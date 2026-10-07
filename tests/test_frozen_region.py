"""The restored regional files must match a frozen inventory, without overwrite."""

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from reproducibility.extract_frozen_region import extract


class FrozenRegionTests(unittest.TestCase):
    def test_extracts_only_verified_manifest_members_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, inventory = root / "data.tar.gz", root / "inventory.json"
            payload = (
                b"fixture payload: extraction tests byte identity, not NetCDF parsing"
            )
            with tarfile.open(archive, "w:gz") as stream:
                for name, value in [
                    ("nested/dominicamaria_s2_1.nc", payload),
                    ("china_s2_2.nc", b"other"),
                ]:
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    stream.addfile(info, io.BytesIO(value))
            inventory.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample": "dominicamaria_s2_1.nc",
                                "region": "dominicamaria",
                                "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ]
                    }
                )
            )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                extract(archive, inventory, "dominicamaria", "0" * 64, root / "bad", 0)
            self.assertFalse((root / "bad").exists())
            result = extract(
                archive, inventory, "dominicamaria", digest, root / "out", 0
            )
            self.assertEqual(result["files"], 1)
            self.assertEqual(
                (root / "out/raw/dominicamaria_s2_1.nc").read_bytes(), payload
            )
            self.assertFalse((root / "out/raw/china_s2_2.nc").exists())
            with self.assertRaises(FileExistsError):
                extract(archive, inventory, "dominicamaria", digest, root / "out", 0)


if __name__ == "__main__":
    unittest.main()
