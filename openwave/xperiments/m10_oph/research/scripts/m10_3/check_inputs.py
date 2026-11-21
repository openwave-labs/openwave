"""Check the three immutable author inputs used by the offline M10.3 replay."""

import hashlib
from pathlib import Path
import sys

INPUTS = {
    "FEDERATION_LEVEL0.npz": "99870398e3d516f899c600e9bf25ba53c7f513142ba8facbe1cedd94b23f915b",
    "row_04_receipt.json": "7f6e458fd0b7cf9b9e50fb15a5c4b77f944a2457a9c53cdcca88efdf4d4c3bdd",
    "row_04_window.json": "f0b594675b4468c35d28954ae754ecf88ab41ad3887449c5fc07bc21566d4754"
}

if __name__ == "__main__":
    root = Path(sys.argv[1])
    for name, digest in INPUTS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Author input changed: {name}")
    print("All three author inputs match.")
