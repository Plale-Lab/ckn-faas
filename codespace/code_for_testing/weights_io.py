# weights_io.py
import json, os, time
from pathlib import Path
from typing import Dict

def load_model_weights(state_path: str, defaults: Dict[str, float]) -> Dict[str, float]:
    p = Path(state_path)
    if not p.exists():
        # first run: return defaults
        return dict(defaults)
    try:
        data = json.loads(p.read_text())
        # merge: ensure every default key exists
        for k, v in defaults.items():
            data.setdefault(k, float(v))
        # ensure floats
        return {k: float(v) for k, v in data.items()}
    except Exception:
        # corrupted? fall back to defaults
        return dict(defaults)

def save_model_weights_atomic(state_path: str, weights: Dict[str, float]) -> None:
    p = Path(state_path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    # ensure directory exists
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(weights, indent=2, sort_keys=True))
    os.replace(tmp, p)  # atomic on same filesystem

def diff_weights(prev: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    keys = set(prev) | set(new)
    return {k: float(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys}
