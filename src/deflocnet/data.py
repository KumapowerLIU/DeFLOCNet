"""Read the existing CSA mixed manifest without unpacking/downloading datasets."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import struct
from array import array
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

from .masks import random_mask

MAX_RECORD_BYTES = 64 * 1024**2


def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for i in range(10):
        if position >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[position]
        position += 1
        if i == 9 and byte > 1:
            raise ValueError("Oversized protobuf varint")
        value |= (byte & 127) << (i * 7)
        if byte < 128:
            return value, position
    raise ValueError("Invalid protobuf varint")


def _fields(data: bytes):
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        field, wire = tag >> 3, tag & 7
        if field < 1:
            raise ValueError("Invalid protobuf field")
        if wire == 0:
            value, position = _varint(data, position)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, position = _varint(data, position)
            else:
                size = 8 if wire == 1 else 4
            end = position + size
            if end > len(data):
                raise ValueError("Truncated protobuf field")
            value, position = data[position:end], end
        else:
            raise ValueError("Unsupported protobuf wire type")
        yield field, value


def _one(data: bytes, number: int) -> bytes:
    values = [value for field, value in _fields(data) if field == number]
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError(f"Expected one protobuf bytes/message field {number}")
    return values[0]


class _BytesOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("TFRecord pickle globals are forbidden")


def decode_tfrecord(payload: bytes) -> bytes:
    features = _one(payload, 1)
    matches = []
    for field, entry in _fields(features):
        if field != 1 or not isinstance(entry, bytes):
            raise ValueError("Invalid TFRecord feature map")
        if _one(entry, 1) == b"state_dict":
            matches.append(_one(_one(_one(entry, 2), 1), 1))
    if len(matches) != 1:
        raise ValueError("Expected exactly one state_dict feature")
    stream = io.BytesIO(matches[0])
    state = _BytesOnlyUnpickler(stream).load()
    if stream.read(1) or type(state) is not dict:
        raise ValueError("Invalid TFRecord state_dict")
    raw = state.get("raw_image")
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_RECORD_BYTES:
        raise ValueError("Missing/bad TFRecord raw_image")
    return raw


class IndexedImages(Dataset):
    def __init__(
        self,
        manifest: str,
        size: int = 256,
        seed: int = 2019,
        training: bool = True,
        expected_sha256: str | None = None,
        min_fraction: float = 0.1,
        max_fraction: float = 0.4,
        rtv_cache: str = "",
        existing_rtv_cache: str = "",
    ):
        self.manifest = str(Path(manifest).resolve())
        self.size, self.seed, self.training = size, seed, training
        self.min_fraction, self.max_fraction = min_fraction, max_fraction
        from .controls import RTVCache
        self.rtv = RTVCache(rtv_cache, existing_rtv_cache) if rtv_cache else None
        self.sources, source_ids, records = [], {}, array("q")
        digest, counts = hashlib.sha256(), Counter()
        self.first_by_kind = {}
        with open(self.manifest, "rb") as stream:
            for number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, list) and len(row) == 3:
                    path, offset, length = row
                    kind, label = 0, "tfrecord"
                elif isinstance(row, dict) and row.get("format") == "tar":
                    path, offset, length = row["path"], row["offset"], row["length"]
                    kind, label = 1, "tar"
                elif isinstance(row, dict) and row.get("format") == "image":
                    path, offset, length = row["path"], 0, -1
                    kind, label = 2, "image"
                else:
                    raise ValueError(f"Invalid manifest record on line {number}")
                if not isinstance(path, str) or not Path(path).is_absolute():
                    raise ValueError("Manifest paths must be absolute")
                if type(offset) is not int or offset < 0 or type(length) is not int:
                    raise ValueError("Invalid record bounds")
                if kind != 2 and not 0 < length <= MAX_RECORD_BYTES + 16:
                    raise ValueError("Invalid record length")
                if path not in source_ids:
                    source_ids[path] = len(self.sources)
                    self.sources.append(path)
                index = len(records) // 4
                self.first_by_kind.setdefault(label, index)
                records.extend((source_ids[path], kind, offset, length))
                counts[label] += 1
        self.records = np.asarray(records, dtype=np.int64).reshape(-1, 4)
        self.sha256 = digest.hexdigest()
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(f"Manifest differs from the CSA snapshot: {self.manifest}")
        if not len(self.records):
            raise ValueError("Empty image manifest")
        self.counts = dict(counts)
        self._handles = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _read(self, index: int) -> bytes:
        source, kind, offset, length = map(int, self.records[index])
        path = self.sources[source]
        if path not in self._handles:
            self._handles[path] = os.open(path, os.O_RDONLY)
            if len(self._handles) > 16:
                _, old = self._handles.popitem(last=False)
                os.close(old)
        fd = self._handles[path]
        self._handles.move_to_end(path)
        total_size = os.fstat(fd).st_size
        if kind == 2:
            length = total_size
        if not 0 < length <= MAX_RECORD_BYTES + 16 or offset + length > total_size:
            raise ValueError(f"Record outside file bounds: {path}:{offset}")
        contents = os.pread(fd, length, offset)
        if len(contents) != length:
            raise ValueError(f"Truncated indexed record: {path}:{offset}")
        if kind == 0:
            if length < 16 or struct.unpack("<Q", contents[:8])[0] + 16 != length:
                raise ValueError("TFRecord framing does not match CSA manifest")
            contents = decode_tfrecord(contents[12:-4])
        return contents

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (int(key), 0)
        if not self.training:
            epoch = 0
        rng = random.Random(self.seed + epoch * 1_000_003 + index * 97)
        with Image.open(io.BytesIO(self._read(index))) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
            array_rgb = np.asarray(image, dtype=np.float32) / 255.0
            structure = self.rtv(array_rgb) if self.rtv else None
            rgb = torch.from_numpy(array_rgb.copy()).permute(2, 0, 1)
            if structure is not None:
                structure = torch.from_numpy(structure.copy()).permute(2, 0, 1)
            if self.training and rng.random() < 0.5:
                rgb = rgb.flip(-1)
                if structure is not None:
                    structure = structure.flip(-1)
        hole = random_mask(self.size, self.size, rng, self.min_fraction, self.max_fraction)
        sample = {"image": rgb, "valid": ~hole, "index": index}
        if structure is not None:
            from .controls import sample_controls
            sample.update(sample_controls(hole, structure, rng))
            sample["rtv"] = structure
        return sample

    def close(self):
        for fd in self._handles.values():
            os.close(fd)
        self._handles.clear()

    def audit(self) -> dict:
        examples = {}
        for kind, index in self.first_by_kind.items():
            sample = self[index]
            examples[kind] = {
                "index": index,
                "shape": list(sample["image"].shape),
                "hole_fraction": float((~sample["valid"]).float().mean()),
            }
        self.close()
        return {
            "manifest": self.manifest,
            "sha256": self.sha256,
            "records": len(self),
            "kinds": self.counts,
            "decoded_examples": examples,
        }


class EpochMicrobatches(Sampler):
    """Resume at a committed cursor, unaffected by DataLoader prefetch."""

    def __init__(self, length: int, batch: int, microbatch: int, epoch: int, seed: int, start: int = 0):
        if batch % microbatch or start % batch:
            raise ValueError("Batch/microbatch or resume cursor is inconsistent")
        self.length = length // batch * batch
        self.microbatch, self.epoch, self.seed, self.start = microbatch, epoch, seed, start
        self.full_length = length

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.full_length, generator=rng)
        for i in range(self.start, self.length, self.microbatch):
            yield [(int(j), self.epoch) for j in order[i : i + self.microbatch]]

    def __len__(self):
        return max(0, (self.length - self.start) // self.microbatch)
