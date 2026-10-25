"""Commit and push the repo from Colab without storing the GitHub token.

The token is read from Colab secrets (key: GITHUB_TOKEN) and passed only
in the push URL, so it never lands in .git/config on Google Drive.
"""

import subprocess

from .utils import REPO_ROOT


def _run(cmd: list[str], secret: str | None = None) -> str:
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if secret:
        text = text.replace(secret, "***")
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} failed:\n{text}")
    return text


def _token() -> str:
    from google.colab import userdata
    return userdata.get("GITHUB_TOKEN")


def _remote_path() -> str:
    """'user/repo.git' from the origin URL."""
    url = _run(["git", "remote", "get-url", "origin"])
    return url.split("github.com/")[-1]


def pull() -> None:
    token = _token()
    url = f"https://{token}@github.com/{_remote_path()}"
    print(_run(["git", "pull", "--rebase", url, "main"], secret=token) or "Up to date")


def push(message: str) -> None:
    """git add -A, commit with `message`, push to main."""
    _run(["git", "add", "-A"])
    status = _run(["git", "status", "--porcelain"])
    if not status:
        print("Nothing new to commit; pushing any earlier commits.")
    else:
        print(status)
        _run(["git", "commit", "-m", message])
    token = _token()
    url = f"https://{token}@github.com/{_remote_path()}"
    _run(["git", "push", url, "HEAD:main"], secret=token)
    print(f"Pushed: {message}")


def save_notebook(repo_path: str) -> None:
    """Write the notebook currently open in Colab to repo_path (e.g. 'notebooks/01_data_eda.ipynb')."""
    import json
    from google.colab import _message
    nb = _message.blocking_request("get_ipynb", timeout_sec=60)["ipynb"]
    out = REPO_ROOT / repo_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved notebook -> {repo_path}")


def save_and_push(repo_path: str, message: str) -> None:
    """Save the open notebook into the repo, then commit and push everything."""
    save_notebook(repo_path)
    push(message)
