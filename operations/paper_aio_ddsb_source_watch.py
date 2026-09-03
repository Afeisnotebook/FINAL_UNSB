"""Durably watch for an authoritative DDSB implementation.

This process is deliberately outside the scientific runner.  It only checks
public authority pages and GitHub repository search.  A hit is *never* a lane
authorization: it writes a review-required state and stops.  A human must then
verify authorship, source identity, formula coverage and full-state semantics
in Git before DDSB can leave ``REPRODUCTION_INCOMPLETE``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen


CONTRACT_SCHEMA = "final-unsb-ddsb-source-watch-contract-v1"
STATE_SCHEMA = "final-unsb-ddsb-source-watch-state-v1"
TITLE = "Degradation-Aware Dynamic Schrodinger Bridge for Unpaired Image Restoration"
DEFAULT_AUTHORITY_URLS = (
    "https://papers.nips.cc/paper_files/paper/2025/hash/"
    "039c30e9af8039fbd1b58da9d04f38e9-Abstract-Conference.html",
    "https://medai.lab.westlake.edu.cn/Publications1.htm",
)
DEFAULT_GITHUB_QUERIES = (
    '"Degradation-Aware Dynamic Schrodinger Bridge" in:name,description,readme',
    'DDSB "unpaired image restoration" in:name,description,readme',
)
REPOSITORY_HOSTS = {"github.com", "www.github.com", "gitlab.com", "www.gitlab.com"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"DDSB source-watch state must be a JSON object: {path}")
    return payload


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", html.unescape(str(value)))
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = dict(attrs)
        self._href = values.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.anchors.append({"href": self._href, "label": " ".join(self._text).strip()})
            self._href = None
            self._text = []


def _title_window(document: str, title: str) -> str:
    lower = document.lower()
    probes = (
        title.lower(),
        "degradation-aware dynamic schr",
        "degradation-aware dynamic schrödinger bridge",
    )
    positions = [lower.find(probe) for probe in probes]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return ""
    position = min(positions)
    return document[max(0, position - 5000): position + 12000]


def authority_repository_candidates(
    document: str, *, authority_url: str, title: str = TITLE,
) -> list[dict[str, str]]:
    """Return repository links in the publication-local title window.

    General lab publication pages can contain many unrelated repositories, so
    repository links outside the local title window are intentionally ignored.
    Every returned link still requires manual authorship review.
    """
    window = _title_window(document, title)
    if not window:
        return []
    parser = _AnchorCollector()
    parser.feed(window)
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in parser.anchors:
        absolute = urljoin(authority_url, anchor["href"])
        host = urlparse(absolute).hostname or ""
        if host.lower() not in REPOSITORY_HOSTS:
            continue
        label = normalized_text(anchor["label"])
        path = normalized_text(urlparse(absolute).path)
        if not any(token in f"{label} {path}" for token in ("code", "github", "gitlab", "repo", "ddsb")):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        candidates.append(
            {"authority_url": authority_url, "repository_url": absolute, "label": anchor["label"]}
        )
    return candidates


def github_repository_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    title_tokens = normalized_text(TITLE)
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        full_name = str(item.get("full_name") or "")
        description = str(item.get("description") or "")
        combined = normalized_text(" ".join((name, full_name, description)))
        exact_title = title_tokens in combined
        concise_identity = "ddsb" in combined and "unpaired image restoration" in combined
        if not (exact_title or concise_identity):
            continue
        owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
        candidates.append(
            {
                "full_name": full_name,
                "html_url": str(item.get("html_url") or ""),
                "description": description,
                "owner_login": str(owner.get("login") or ""),
                "exact_title_match": exact_title,
                "concise_identity_match": concise_identity,
            }
        )
    return candidates


def fetch_bytes(url: str, *, timeout_seconds: float = 30.0) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json,text/html;q=0.9,*/*;q=0.8",
            "User-Agent": "FINAL-UNSB-DDSB-source-watch/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read()


def evaluate_sources(
    contract: dict[str, Any], *,
    fetcher: Callable[..., bytes] = fetch_bytes,
    now: float | None = None,
) -> dict[str, Any]:
    authority_checks: list[dict[str, Any]] = []
    official_candidates: list[dict[str, str]] = []
    github_checks: list[dict[str, Any]] = []
    unverified_candidates: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for url in contract["authority_urls"]:
        try:
            raw = fetcher(url, timeout_seconds=contract["request_timeout_seconds"])
            document = raw.decode("utf-8", errors="replace")
            candidates = authority_repository_candidates(document, authority_url=url)
            official_candidates.extend(candidates)
            authority_checks.append(
                {"url": url, "response_sha256": bytes_sha256(raw), "repository_candidates": candidates}
            )
        except Exception as error:  # network errors are state, not scientific failure
            errors.append({"source": url, "error_type": type(error).__name__, "error": str(error)})

    for query in contract["github_queries"]:
        url = "https://api.github.com/search/repositories?q=" + quote_plus(query)
        try:
            raw = fetcher(url, timeout_seconds=contract["request_timeout_seconds"])
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("GitHub search response is not an object")
            candidates = github_repository_candidates(payload)
            unverified_candidates.extend(candidates)
            github_checks.append(
                {
                    "query": query,
                    "response_sha256": bytes_sha256(raw),
                    "total_count": int(payload.get("total_count", 0)),
                    "relevant_candidates": candidates,
                }
            )
        except Exception as error:
            errors.append({"source": url, "error_type": type(error).__name__, "error": str(error)})

    if official_candidates:
        status = "AUTHORITATIVE_SOURCE_CANDIDATE_REVIEW_REQUIRED"
    elif unverified_candidates:
        status = "UNVERIFIED_SOURCE_CANDIDATE_REVIEW_REQUIRED"
    elif errors and not authority_checks and not github_checks:
        status = "TRANSIENT_NETWORK_ERROR"
    else:
        status = "WAITING_FOR_AUTHORITATIVE_SOURCE"
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "checked_unix_time": time.time() if now is None else float(now),
        "title": TITLE,
        "authority_checks": authority_checks,
        "github_checks": github_checks,
        "official_repository_candidates": official_candidates,
        "unverified_repository_candidates": unverified_candidates,
        "network_errors": errors,
        "paper_status": "REPRODUCTION_INCOMPLETE",
        "manual_formula_source_implementation_review_required": True,
        "training_authorized": False,
        "training_started": False,
        "checkpoint_loaded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authority-url", action="append")
    parser.add_argument("--github-query", action="append")
    parser.add_argument("--poll-seconds", type=int, default=21600)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def proposed_contract(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output.is_absolute():
        raise ValueError("DDSB source-watch output must be absolute")
    if int(args.poll_seconds) < 3600 or int(args.poll_seconds) > 86400:
        raise ValueError("DDSB source-watch poll interval must be in [3600,86400]")
    if float(args.timeout_hours) < 24:
        raise ValueError("DDSB source-watch timeout must be at least 24 hours")
    if not 1 <= float(args.request_timeout_seconds) <= 120:
        raise ValueError("DDSB source-watch request timeout must be in [1,120]")
    authority_urls = tuple(args.authority_url or DEFAULT_AUTHORITY_URLS)
    github_queries = tuple(args.github_query or DEFAULT_GITHUB_QUERIES)
    if not authority_urls or not github_queries:
        raise ValueError("DDSB source-watch needs authority URLs and GitHub queries")
    if any(urlparse(url).scheme != "https" for url in authority_urls):
        raise ValueError("DDSB authority URLs must use HTTPS")
    script = Path(__file__).resolve()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_FAIL_CLOSED",
        "title": TITLE,
        "control_script": str(script),
        "control_script_sha256": file_sha256(script),
        "output": str(args.output.resolve()),
        "authority_urls": list(authority_urls),
        "github_queries": list(github_queries),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "request_timeout_seconds": float(args.request_timeout_seconds),
        "automatic_source_acceptance": False,
        "automatic_training_authorization": False,
        "performance_values_available": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def freeze_contract(output: Path, proposed: dict[str, Any]) -> Path:
    path = Path(output).resolve() / "DDSB_SOURCE_WATCH_CONTRACT.json"
    if path.is_file():
        if read_json(path) != proposed:
            raise RuntimeError("DDSB source-watch contract changed")
        return path
    atomic_json(path, proposed)
    return path


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(query_limited_information, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(output: Path) -> Path:
    path = Path(output).resolve() / "DDSB_SOURCE_WATCH.lock"
    if path.is_file():
        try:
            current = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            current = 0
        if current > 0 and process_alive(current):
            raise RuntimeError(f"DDSB source watcher already running: pid={current}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    return path


def main() -> int:
    args = arguments()
    output = args.output.resolve()
    contract = proposed_contract(args)
    freeze_contract(output, contract)
    if args.once:
        atomic_json(output / "DDSB_SOURCE_WATCH_STATE.json", evaluate_sources(contract))
        return 0

    lock = acquire_lock(output)
    started = time.time()
    deadline = started + float(contract["timeout_hours"]) * 3600.0
    try:
        while True:
            if file_sha256(Path(contract["control_script"])) != contract["control_script_sha256"]:
                raise RuntimeError("DDSB source-watch control script changed")
            state = evaluate_sources(contract)
            state["watcher_pid"] = os.getpid()
            state["watch_started_unix_time"] = started
            atomic_json(output / "DDSB_SOURCE_WATCH_STATE.json", state)
            if state["status"].endswith("REVIEW_REQUIRED"):
                return 0
            if time.time() >= deadline:
                state["status"] = "TIMED_OUT_WITHOUT_AUTHORITY_DECISION"
                atomic_json(output / "DDSB_SOURCE_WATCH_STATE.json", state)
                return 3
            time.sleep(int(contract["poll_seconds"]))
    except Exception as error:
        atomic_json(
            output / "DDSB_SOURCE_WATCH_FATAL.json",
            {
                "schema": STATE_SCHEMA,
                "status": "FATAL",
                "error_type": type(error).__name__,
                "error": str(error),
                "training_authorized": False,
                "training_started": False,
                "performance_values_read": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            },
        )
        return 4
    finally:
        if lock.is_file() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
