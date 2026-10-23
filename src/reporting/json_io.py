"""Write full-precision JSON reports with NumPy values."""

import json
from pathlib import Path

import numpy as np


def _json_default(value):
    """Convert NumPy arrays/scalars without rounding their numeric values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON.")


def save_json_report(report, output_path):
    """Create parent directories, replace the JSON file, and return its Path."""
    # Serialize before opening the file, so invalid values preserve prior output.
    serialized = json.dumps(
        report, indent=2, default=_json_default, allow_nan=False, ensure_ascii=False
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return output_path
