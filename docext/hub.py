"""Share results through the Hugging Face Hub, so anyone can re-score runs,
redo routing and rebuild the result tables without a GPU.

    python -m docext.hub push <user>/doc-extraction-results   # needs a write token (huggingface-cli login)
    python -m docext.hub pull <user>/doc-extraction-results   # a public repo needs no account

Shared: run folders (config, history, predictions, metrics, routing), logs,
dataset reports and result tables. Not shared: fine-tuned weights and the
converted datasets, which anyone can rebuild with docext.prepare and which,
for DocILE, its terms do not allow redistributing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

SHARED = ["runs/**", "logs/**", "data/*/*.json", "results_*.csv"]
NOT_SHARED = ["runs/*/model/**"]


def push(repo_id: str, root: Path, private: bool = False) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        allow_patterns=SHARED,
        ignore_patterns=NOT_SHARED,
        commit_message="Update results",
    )
    print(f"pushed to https://huggingface.co/datasets/{repo_id}")


def pull(repo_id: str, root: Path) -> None:
    try:
        snapshot_download(repo_id, repo_type="dataset", local_dir=root, allow_patterns=SHARED)
        print(f"restored results from {repo_id}")
    except RepositoryNotFoundError:
        print(f"{repo_id} does not exist yet; nothing to restore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("repo_id", help="Hugging Face dataset repo, e.g. user/doc-extraction-results")
    parser.add_argument("--root", type=Path, default=Path("."), help="project folder")
    parser.add_argument("--private", action="store_true", help="create the repo as private (push only)")
    args = parser.parse_args()
    if args.action == "push":
        push(args.repo_id, args.root, args.private)
    else:
        pull(args.repo_id, args.root)


if __name__ == "__main__":
    main()
