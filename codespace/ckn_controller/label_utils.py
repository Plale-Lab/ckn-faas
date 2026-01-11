import json, re
from pathlib import Path

LABELS_JSON_PATH = Path("Labels.json")

def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)

def _tokenize(s: str):
    return re.findall(r"[a-z]+", (s or "").lower())

def load_synsets(path: Path):
    raw = json.loads(path.read_text())
    wnid_to_names, wnid_to_canonical = {}, {}
    for wnid, names in raw.items():
        parts = [_normalize_label(p) for p in str(names).split(",")]
        parts = [p for p in (x.strip() for x in parts) if p]
        if not parts:
            continue
        wnid_to_names[wnid] = set(parts)
        wnid_to_canonical[wnid] = parts[0]
    return wnid_to_names, wnid_to_canonical

# Load once at import
WNID_TO_NAMES, WNID_TO_CANON = load_synsets(LABELS_JSON_PATH)

def wnid_matches_text_label(wnid: str, pred_label: str) -> bool:
    """Check if predicted label matches any synonym for the given WNID."""
    if not wnid or not pred_label:
        return False
    syns = WNID_TO_NAMES.get(wnid, set())
    if not syns:
        return False
    pt = set(_tokenize(pred_label))
    if not pt:
        return False
    for s in syns:
        if pt == set(_tokenize(s)) or pt.issubset(set(_tokenize(s))):
            return True
    return False


with open("Labels.json") as f:
    WNID_TO_CANON = json.load(f)
def name_from_wnid(wnid):
    return WNID_TO_CANON.get(wnid, wnid)

