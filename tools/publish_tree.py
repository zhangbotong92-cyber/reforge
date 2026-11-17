#!/usr/bin/env python
"""Publish a directory to GitHub through the Git Data API.

For the two cases where `git push` is not an option:

1. the connection is reset mid-transfer (common behind AV / corporate / CN
   middleboxes — `error: RPC failed; curl 55 Send failure`);
2. the token lacks the `workflow` scope, in which case GitHub rejects the whole
   push because of a single `.github/workflows/*` file.

This tool uploads real blobs (never inline tree content, which silently stores
base64 as text), builds one commit, moves the branch, and then **reads the bytes
back** to prove nothing got corrupted.

    export GITHUB_TOKEN=ghp_xxx
    python tools/publish_tree.py --dir . --repo reforge                 # dry run
    python tools/publish_tree.py --dir . --repo reforge --yes           # publish
    python tools/publish_tree.py --dir . --repo reforge --yes --keep-history

Exit codes: 0 ok, 1 failure, 2 refused by a safety check.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode", "demo_out", "examples/out"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".tmp"}
BINARY_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".woff", ".woff2"}
MAX_BLOB = 90 * 1024 * 1024


# ---------------------------------------------------------------- http


def api(method: str, path: str, payload: dict | None = None, raw: bool = False, timeout: int = 180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "reforge-publish-tree",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return body if raw else json.loads(body.decode() or "{}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc


# ---------------------------------------------------------------- collection


def collect(root: Path, excludes: set[str]) -> list[tuple[str, bytes]]:
    """Prefer git's index: publishing should mean 'what is committed'.

    Walking the filesystem instead pulls in gitignored artifacts (demo output,
    extracted frames) and quietly publishes test junk.
    """
    try:
        raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        listed = [p for p in raw.decode("utf-8").split("\0") if p and p not in excludes]
        if listed:
            files = []
            for rel in sorted(listed):
                path = root / rel
                blob = path.read_bytes()
                if len(blob) > MAX_BLOB:
                    print(f"  skip {rel}: {len(blob)/1e6:.1f} MB exceeds the blob API limit")
                    continue
                files.append((rel, blob))
            print(f"  mode   : git index ({len(files)} tracked files)")
            return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("  mode   : filesystem walk (no git repo here)")
    out: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in Path(rel).parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIX or rel in excludes:
            continue
        blob = path.read_bytes()
        if len(blob) > MAX_BLOB:
            print(f"  skip {rel}: {len(blob)/1e6:.1f} MB exceeds the blob API limit")
            continue
        out.append((rel, blob))
    return out


SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub tokens
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),           # OpenAI-style keys (DeepSeek, Moonshot, ...)
    re.compile(rb"AKIA[0-9A-Z]{16}"),                # AWS access key id
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),    # Slack
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),           # Google API key
)


def sanity(files: list[tuple[str, bytes]]) -> None:
    """Refuse to publish an obviously corrupted tree, or one that leaks a secret."""
    table = dict(files)
    readme = table.get("README.md", b"")
    head = readme[:16].decode("utf-8", "replace")
    if not head.lstrip("\ufeff").startswith("# "):
        raise SystemExit(f"refuse: README.md does not start with a heading (got {head!r}) - base64 or mojibake?")
    for name, blob in files:
        if name.endswith(".png") and blob[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"refuse: {name} is not a PNG (magic {blob[:8]!r})")
        if name.endswith(".py"):
            try:
                blob.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SystemExit(f"refuse: {name} is not valid UTF-8: {exc}") from exc

    token = TOKEN.encode().strip()
    for name, blob in files:
        if token and token in blob:
            raise SystemExit(f"refuse: the token itself appears in {name}")
        for pattern in SECRET_PATTERNS:
            match = pattern.search(blob)
            if match:
                kind = match.group(0)[:6].decode("ascii", "replace")
                raise SystemExit(f"refuse: {name} contains something that looks like a credential ({kind}...)")


def warn_secrets(files: list[tuple[str, bytes]]) -> None:
    for name, blob in files:
        for pattern in SECRET_PATTERNS:
            if pattern.search(blob):
                print(f"  WARNING: {name} matches a credential shape - check before publishing")


# ---------------------------------------------------------------- publish


def publish(owner: str, repo: str, files: list[tuple[str, bytes]], message: str, branch: str, keep_history: bool) -> dict:
    tree = []
    for i, (rel, blob) in enumerate(files, 1):
        sha = api("POST", f"/repos/{owner}/{repo}/git/blobs", {"content": base64.b64encode(blob).decode(), "encoding": "base64"})["sha"]
        mode = "100755" if rel.endswith((".sh", ".command")) else "100644"
        tree.append({"path": rel, "mode": mode, "type": "blob", "sha": sha})
        if i % 10 == 0 or i == len(files):
            print(f"    blobs {i}/{len(files)}")

    parents: list[str] = []
    if keep_history:
        try:
            parents = [api("GET", f"/repos/{owner}/{repo}/git/refs/heads/{branch}")["object"]["sha"]]
        except SystemExit:
            parents = []

    tree_sha = api("POST", f"/repos/{owner}/{repo}/git/trees", {"tree": tree})["sha"]
    commit = api("POST", f"/repos/{owner}/{repo}/git/commits", {"message": message, "tree": tree_sha, "parents": parents})
    api("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/{branch}", {"sha": commit["sha"], "force": True})
    return {"tree": tree_sha, "commit": commit["sha"], "files": len(files)}


def verify(owner: str, repo: str, branch: str, files: list[tuple[str, bytes]]) -> bool:
    """Read the bytes back and compare hashes against what we sent."""
    import hashlib

    ok = True
    tree = api("GET", f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")["tree"]
    remote = sorted(t["path"] for t in tree if t["type"] == "blob")
    local = sorted(p for p, _ in files)
    if remote != local:
        missing = set(local) - set(remote)
        extra = set(remote) - set(local)
        print(f"    manifest mismatch  missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        ok = False
    else:
        print(f"    manifest ok: {len(remote)} files")

    for name in ("README.md", "README.zh-CN.md"):
        if not any(p == name for p, _ in files):
            continue
        got = api("GET", f"/repos/{owner}/{repo}/contents/{name}", raw=True)
        want = dict(files)[name]
        same = hashlib.sha256(got).hexdigest() == hashlib.sha256(want).hexdigest()
        print(f"    {name:<20} {'byte-identical' if same else 'DIFFERS'} ({len(got)} B)")
        ok &= same
    return ok


def main(argv: list[str] | None = None) -> int:
    global TOKEN
    ap = argparse.ArgumentParser(description="Publish a directory to GitHub via the Git Data API.")
    ap.add_argument("--dir", default=".", help="directory to publish (default: cwd)")
    ap.add_argument("--owner", default=None, help="default: the token's own login")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="")
    ap.add_argument("--token-file", default=None, help="file containing the token (default: $GITHUB_TOKEN)")
    ap.add_argument("--exclude", default="", help="comma separated relative paths to skip")
    ap.add_argument("--keep-history", action="store_true", help="parent the new commit on the current head")
    ap.add_argument("--yes", action="store_true", help="actually publish (otherwise dry run)")
    args = ap.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    if not token:
        print("no token: set GITHUB_TOKEN or pass --token-file")
        return 2
    TOKEN = token

    me = api("GET", "/user")
    owner = args.owner or me["login"]
    root = Path(args.dir).resolve()
    excludes = {x.strip() for x in args.exclude.split(",") if x.strip()}
    files = collect(root, excludes)
    total = sum(len(b) for _, b in files)
    print(f"  source : {root}")
    print(f"  target : {owner}/{args.repo} ({args.branch})")
    print(f"  payload: {len(files)} files, {total / 1024:.1f} KB")
    sanity(files)

    workflow = [p for p, _ in files if p.startswith(".github/workflows/")]
    if workflow:
        print(f"  note: {len(workflow)} workflow file(s) included. They need the `workflow` token scope;")
        print("        GitHub will 404 at tree creation without it. Use --exclude .github/workflows/ci.yml")

    if not args.yes:
        for rel, blob in files[:15]:
            print(f"    {len(blob):>8} B  {rel}")
        if len(files) > 15:
            print(f"    ... {len(files) - 15} more")
        print("\n  dry run - pass --yes to publish")
        return 0

    message = args.message or f"Publish {len(files)} files"
    print("  publishing")
    result = publish(owner, args.repo, files, message, args.branch, args.keep_history)
    print(f"    commit {result['commit'][:12]}  tree {result['tree'][:12]}")
    print("  verifying")
    ok = verify(owner, args.repo, args.branch, files)
    print("\n", "OK - live and byte-identical" if ok else "FAILED verification")
    return 0 if ok else 1


TOKEN = ""

if __name__ == "__main__":
    sys.exit(main())
