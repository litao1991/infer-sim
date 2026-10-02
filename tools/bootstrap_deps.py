#!/usr/bin/env python3
"""Fetch pinned simulator dependencies into .deps and apply the recorded patch.

Existing checkouts at another revision are rejected; user changes are never reset.
Requires Git and network access on the first run. No model weights are downloaded.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(folder, *args, check=True, capture=False):
    return subprocess.run(['git', '-C', str(folder), *args], check=check,
                          text=True, capture_output=capture)


def main():
    manifest = json.loads((ROOT / 'dependencies.lock.json').read_text())
    destination = ROOT / '.deps'
    destination.mkdir(exist_ok=True)
    for name, spec in manifest.items():
        checkout = destination / name
        if not checkout.exists():
            subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(checkout)], check=True)
            git(checkout, 'checkout', '--detach', spec['revision'])
        top = Path(git(checkout, 'rev-parse', '--show-toplevel', capture=True).stdout.strip())
        if top.resolve() != checkout.resolve():
            raise ValueError(f'{checkout} is not a standalone dependency checkout')
        actual = git(checkout, 'rev-parse', 'HEAD', capture=True).stdout.strip()
        if actual != spec['revision']:
            raise ValueError(f'{name}: expected {spec["revision"]}, found {actual}; preserve or move this checkout before retrying')
        if 'patch' in spec:
            patch = ROOT / spec['patch']
            if hashlib.sha256(patch.read_bytes()).hexdigest() != spec['patch_sha256']:
                raise ValueError(f'{name}: patch checksum mismatch')
            already = git(checkout, 'apply', '--reverse', '--check', str(patch), check=False, capture=True)
            if already.returncode:
                git(checkout, 'apply', '--check', str(patch))
                git(checkout, 'apply', str(patch))
        print(f'{name}: {actual}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
