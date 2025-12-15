from pathlib import Path
def ensure_case_folder(root: str, case_number: str) -> str:
    safe = case_number.replace('/', '-').replace('\\', '-').replace(' ', '_')
    d = Path(root) / safe
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def compute_offer_70(arv: float, rehab: float, closing: float) -> float:
    try:
        return max(0.0, (float(arv) - float(rehab) - float(closing)) * 0.70)
    except Exception:
        return 0.0