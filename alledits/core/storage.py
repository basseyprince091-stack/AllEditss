"""Object storage abstraction.

Principle 11: raw media, proxies and outputs are kept in separate namespaces.
Principle 12: source files are copied in and never mutated.

LocalStorage is the day-one implementation. S3Storage can be dropped in later
without touching callers — the interface is deliberately small.
"""
from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

RAW = "raw"
PROXY = "proxy"
ANALYSIS = "analysis"
RENDER = "render"
OUTPUT = "output"
NAMESPACES = (RAW, PROXY, ANALYSIS, RENDER, OUTPUT)


class Storage(ABC):
    @abstractmethod
    def path(self, namespace: str, key: str) -> Path: ...

    @abstractmethod
    def put_file(self, namespace: str, key: str, src: Path) -> Path: ...

    @abstractmethod
    def exists(self, namespace: str, key: str) -> bool: ...

    def put_json(self, namespace: str, key: str, obj) -> Path:
        p = self.path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def get_json(self, namespace: str, key: str):
        p = self.path(namespace, key)
        if not p.exists():
            return None
        return json.loads(p.read_text())


class LocalStorage(Storage):
    def __init__(self, root: Path | str):
        self.root = Path(root)
        for ns in NAMESPACES:
            (self.root / ns).mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str) -> Path:
        assert namespace in NAMESPACES, f"unknown namespace {namespace}"
        return self.root / namespace / key

    def exists(self, namespace: str, key: str) -> bool:
        return self.path(namespace, key).exists()

    def put_file(self, namespace: str, key: str, src: Path) -> Path:
        dst = self.path(namespace, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).resolve() != dst.resolve():
            shutil.copy2(src, dst)   # copy, never move: preserve user source
        return dst
