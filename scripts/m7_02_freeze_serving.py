"""
Module 7, Step 3a: write the exact package versions the API needs into docker/ (run once before `docker compose build`).

    uv run python scripts/m7_02_freeze_serving.py

Why exact versions: the trained models were saved with YOUR versions of scikit-learn, xgboost etc. A different
version inside the container can fail to load them, or silently predict differently. Pinning removes that risk.
torch goes in a separate file because the container installs the small CPU-only build (no GPU inside Docker).
"""
import sys
from importlib.metadata import PackageNotFoundError, version

from cfpb import config

SERVING = ["fastapi", "uvicorn", "pydantic", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib",
           "xgboost", "requests", "faiss-cpu", "rank-bm25", "sentence-transformers", "transformers",
           "huggingface-hub", "tokenizers", "safetensors"]


def main():
    lines, missing = [], []
    for pkg in SERVING:
        try:
            lines.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            missing.append(pkg)
    out = config.ROOT / "docker"
    out.mkdir(exist_ok=True)
    (out / "requirements-serve.txt").write_text("\n".join(lines) + "\n")
    torch_v = version("torch").split("+")[0]
    (out / "torch.txt").write_text(f"torch=={torch_v}\n")
    print("\n".join(lines) + f"\ntorch=={torch_v}  (CPU build in the container)")
    print(f"\nYour Python: {sys.version_info.major}.{sys.version_info.minor} (the container uses 3.12)")
    if missing:
        raise SystemExit(f"Not installed here, so not pinned: {missing}. If the API needs them, `uv add` them first.")
    print(f"Saved docker/requirements-serve.txt and docker/torch.txt. Next: docker compose up --build")


if __name__ == "__main__":
    main()
