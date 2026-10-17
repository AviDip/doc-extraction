"""Run the experiments listed in jobs.json and publish results to GitHub.

Started once (e.g. in Colab); after that, experiments are controlled by
editing jobs.json on the main branch. Each cycle the runner pulls the latest
code, starts jobs whose dependencies have finished, and after every job
pushes logs, run folders and a status file to the `results` branch.

    GITHUB_TOKEN=... python -m docext.runner --repo AviDip/doc-extraction

jobs.json:
    {"parallel": 3,
     "jobs": [{"id": "cord-data", "cmd": "python -m docext.prepare cord --out data"},
              {"id": "cord-split", "cmd": "...", "needs": ["cord-data"]}]}

"creates" names a file a job produces that is not published (a converted
dataset); if it is missing after a restart, the job runs again. Raising a
job's "rev" (default 0) reruns it, e.g. after a fix.

Only commands starting with "python -m docext." or "bash scripts/" run, so
the job list can never execute anything outside this project.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

ALLOWED = ("python -m docext.", "bash scripts/")
SECRET_VARS = ("GITHUB_TOKEN", "DOCILE_TOKEN")  # never published, even if a tool prints them
POLL_SECONDS = 60
RESULTS_BRANCH = "results"


def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run git; on failure raise with git's own message, token removed."""
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode:
        message = re.sub(r"x-access-token:[^@]+@", "x-access-token:***@", proc.stderr.strip())
        raise RuntimeError(f"git {args[0]} failed: {message}")
    return proc


class Results:
    """A clone of the results branch, used to restore and publish outputs."""

    def __init__(self, remote: str, path: Path):
        self.path = path
        if not path.exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1", remote, str(path)], check=True)
            git("config", "user.name", "docext-runner", cwd=path)
            git("config", "user.email", "runner@users.noreply.github.com", cwd=path)
            if git("fetch", "-q", "--depth", "1", "origin", RESULTS_BRANCH, cwd=path, check=False).returncode == 0:
                git("checkout", "-q", "-B", RESULTS_BRANCH, "FETCH_HEAD", cwd=path)
            else:  # first run: an empty branch with no code in it
                git("checkout", "-q", "--orphan", RESULTS_BRANCH, cwd=path)
                git("rm", "-rq", "--cached", ".", cwd=path, check=False)
                for item in path.iterdir():
                    if item.name != ".git":
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
                git("commit", "-q", "--allow-empty", "-m", "Start results branch", cwd=path)
        try:  # fail now, before any job starts, if the token cannot push
            git("push", "-q", "-u", "origin", RESULTS_BRANCH, cwd=path)
        except RuntimeError as err:
            raise SystemExit(
                f"{err}\nGitHub refused the push. Check that the token is fine-grained, limited to this "
                "repository, and has 'Contents: Read and write' permission."
            ) from None

    def status(self) -> dict:
        f = self.path / "status.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def restore(self, root: Path) -> None:
        """Copy published outputs back, so finished runs are skipped after a restart."""
        for name in ("runs", "logs", "data"):
            if (self.path / name).exists():
                shutil.copytree(self.path / name, root / name, dirs_exist_ok=True)

    def publish(self, root: Path, status: dict, message: str) -> None:
        for name in ("runs", "logs"):
            if (root / name).exists():
                shutil.copytree(root / name, self.path / name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("model"))
        self._redact_logs()
        for report in list((root / "data").glob("*/*.json")) + list(root.glob("results_*.csv")):
            target = self.path / report.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report, target)
        (self.path / "status.json").write_text(json.dumps(status, indent=2))
        git("add", "-A", cwd=self.path)
        if git("diff", "--cached", "--quiet", cwd=self.path, check=False).returncode:
            git("commit", "-q", "-m", message, cwd=self.path)
        try:  # a failed push is retried with the next publish; jobs keep running
            git("push", "-q", "origin", RESULTS_BRANCH, cwd=self.path)
        except RuntimeError as err:
            print(f"warning: {err}")

    def _redact_logs(self) -> None:
        secrets = [os.environ[v] for v in SECRET_VARS if os.environ.get(v)]
        for log in (self.path / "logs").glob("*.log") if secrets else []:
            text = log.read_text(errors="replace")
            clean = text
            for secret in secrets:
                clean = clean.replace(secret, "***")
            if clean != text:
                log.write_text(clean)


def load_jobs(root: Path, last: dict) -> dict:
    try:
        return json.loads((root / "jobs.json").read_text())
    except (OSError, json.JSONDecodeError) as err:
        print(f"jobs.json unreadable ({err}); keeping the previous list")
        return last


def start(job: dict, root: Path) -> subprocess.Popen:
    if not job["cmd"].startswith(ALLOWED):
        raise ValueError(f"refusing {job['cmd']!r}: commands must start with {ALLOWED}")
    (root / "logs").mkdir(exist_ok=True)
    log = open(root / "logs" / f"{job['id']}.log", "w")
    env = os.environ | {"PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(shlex.split(job["cmd"]), cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env)


def run(remote: str, root: Path, results_path: Path) -> None:
    results = Results(remote, results_path)
    results.restore(root)
    status = results.status()
    # jobs that were running when the previous session died are retried
    status = {k: v for k, v in status.items() if v["state"] != "running"}
    running: dict[str, tuple[subprocess.Popen, float]] = {}
    jobs: dict = {"parallel": 1, "jobs": []}

    while True:
        git("pull", "-q", "--ff-only", cwd=root, check=False)
        jobs = load_jobs(root, jobs)

        for job_id, (proc, began) in list(running.items()):
            if proc.poll() is not None:
                state = "done" if proc.returncode == 0 else "failed"
                status[job_id] |= {"state": state, "returncode": proc.returncode, "minutes": round((time.time() - began) / 60, 1)}
                del running[job_id]
                print(f"{state}: {job_id}")
                results.publish(root, status, f"{state}: {job_id}")

        for job in jobs["jobs"]:  # outputs not published (datasets) are rebuilt after a restart
            if "creates" in job and status.get(job["id"], {}).get("state") == "done" and not (root / job["creates"]).exists():
                del status[job["id"]]
        done = {k for k, v in status.items() if v["state"] == "done"}
        for job in jobs["jobs"]:
            if len(running) >= jobs.get("parallel", 1):
                break
            previous = status.get(job["id"])
            if job["id"] in running or not set(job.get("needs", [])) <= done:
                continue
            if previous and previous.get("rev", 0) == job.get("rev", 0):
                continue
            try:
                running[job["id"]] = (start(job, root), time.time())
            except ValueError as err:
                status[job["id"]] = {"state": "failed", "error": str(err), "rev": job.get("rev", 0)}
                continue
            status[job["id"]] = {"state": "running", "rev": job.get("rev", 0)}
            print(f"started: {job['id']}")
            results.publish(root, status, f"started: {job['id']}")

        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name of the GitHub repository")
    parser.add_argument("--root", type=Path, default=Path("."), help="project folder (a clone of main)")
    parser.add_argument("--results", type=Path, default=Path("../results"), help="where to clone the results branch")
    args = parser.parse_args()
    remote = f"https://x-access-token:{os.environ['GITHUB_TOKEN']}@github.com/{args.repo}.git"
    run(remote, args.root.resolve(), args.results.resolve())


if __name__ == "__main__":
    main()
