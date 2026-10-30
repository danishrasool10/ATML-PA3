"""Download hub checkpoints once (with retries) before worker processes start."""
from __future__ import annotations

from pa3common.kaggle import retry


def main() -> None:
    from huggingface_hub import snapshot_download

    for rid, pats in (("Qwen/Qwen3-0.6B", ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"]),
                      ("google/siglip-base-patch16-224", ["*.json", "*.safetensors", "*.model", "*.txt"])):
        p = retry(lambda: snapshot_download(rid, allow_patterns=pats), 4, 20, f"snapshot {rid}")
        print(f"[prefetch] {rid} -> {p}", flush=True)


if __name__ == "__main__":
    main()
