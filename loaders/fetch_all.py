from __future__ import annotations

from loaders import biggen, helm, wildbench


def main() -> None:
    print("=== HELM Instruct (GCS bucket) ===")
    helm.fetch_all()
    print()
    print("=== BiGGen-Bench (HuggingFace Hub) ===")
    biggen.fetch_all()
    print()
    print("=== WildBench v2.0522 (allenai/WildBench GitHub) ===")
    wildbench.fetch_all()


if __name__ == "__main__":
    main()
