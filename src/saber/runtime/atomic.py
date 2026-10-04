"""Filesystem helpers: atomic writes and content digests.

Reports are written through a temporary file in the destination directory and
renamed, so a reader never observes a half-written JSON document.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        # mkstemp creates 0600; the release's artefacts are world-readable.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True)
    _atomic_write(path, (text + "\n").encode("utf-8"))


def write_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def write_bytes(path: Path, payload: bytes) -> None:
    _atomic_write(path, payload)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tensors(path: Path, tensors: dict[str, Any]) -> None:
    """Save a tensor payload with a content digest beside it.

    torch.save embeds container metadata that changes on every write, so the
    digest of the shard file is not a function of the payload. Digest the
    payload instead and keep the shard as a transport.
    """
    import torch

    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        array = value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else value
        digest.update(name.encode())
        digest.update(np_as_bytes(array))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)
    write_text(path.with_suffix(path.suffix + ".sha256"), digest.hexdigest() + "\n")


def np_as_bytes(array: Any) -> bytes:
    import numpy as np

    return np.ascontiguousarray(np.asarray(array)).tobytes()


def load_tensors(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not hold a tensor mapping")
    return payload
