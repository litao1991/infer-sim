#!/usr/bin/env python3
"""Download and verify the pinned public Qwen-Bailian usage traces."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

if __package__:
    from .bailian_eval import FILES, ROOT, SOURCE_REPOSITORY, SOURCE_REVISION
else:
    from bailian_eval import FILES, ROOT, SOURCE_REPOSITORY, SOURCE_REVISION

EXPECTED = {
    "qwen_traceA_blksz_16.jsonl": (56354493, "07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17"),
    "qwen_traceB_blksz_16.jsonl": (96209982, "68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59"),
    "qwen_coder_blksz_16.jsonl": (132054902, "3d74974cebb7bccb69b1ca6ea210a44d9a3033443c0271ed015327284dea1aff"),
    "qwen_thinking_blksz_16.jsonl": (27901454, "41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906"),
}


def verify(path, name):
    expected_bytes, expected_hash = EXPECTED[name]
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if (size, digest.hexdigest()) != (expected_bytes, expected_hash):
        raise ValueError(f"checksum or size mismatch: {path}")
    return {"file": name, "bytes": size, "sha256": digest.hexdigest()}


def download(data_dir, sources=FILES, verify_only=False):
    data_dir = Path(data_dir)
    if not verify_only:
        data_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in sources:
        if name not in EXPECTED:
            raise ValueError(f"unknown trace file: {name}")
        path = data_dir / name
        if path.exists():
            results.append(verify(path, name))
            continue
        if verify_only:
            raise FileNotFoundError(path)
        url = ("https://media.githubusercontent.com/media/alibaba-edu/"
               f"qwen-bailian-usagetraces-anon/{SOURCE_REVISION}/{name}")
        pending = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "infer-sim-trace-eval/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as source, pending.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
            result = verify(pending, name)
            pending.replace(path)
            results.append(result)
        finally:
            pending.unlink(missing_ok=True)
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "files": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data/bailian"))
    parser.add_argument("--sources", nargs="+", choices=FILES, default=list(FILES))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(download(args.data_dir, args.sources, args.verify_only), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
