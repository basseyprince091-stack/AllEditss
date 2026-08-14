"""Stable, human-legible identifiers."""
import hashlib
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def content_id(prefix: str, *parts) -> str:
    """Deterministic id from content — enables analysis caching (Principle 10)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return f"{prefix}_{h.hexdigest()[:16]}"


def file_fingerprint(path, chunk=1 << 20) -> str:
    """Hash size + head + tail. Fast for large media, sufficient to key a cache."""
    import os
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:24]
