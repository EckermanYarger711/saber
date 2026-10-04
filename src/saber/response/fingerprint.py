"""Hashed molecular fingerprints.

Ref: Sec. 2.6 -- the drug representation ``u`` is "created using a molecular graph
and a hashed fingerprint". The fingerprint here is the circular
substructure-hashing construction: each atom's environment is folded into a bit
index by iterating a hash over its neighbours' current codes, and each code sets
one bit. The construction is deterministic in the molecule and the radius, which
is what lets two corpora's representations be pooled without a shared
dictionary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

ATOM_CODES: dict[str, int] = {
    "C": 1,
    "N": 2,
    "O": 3,
    "S": 4,
    "F": 5,
    "Cl": 6,
    "Br": 7,
    "P": 8,
}


@dataclass(frozen=True)
class Molecule:
    """A small molecule: ordered atom symbols and ordered bonds."""

    atoms: tuple[str, ...]
    bonds: tuple[tuple[int, int, int], ...]

    def neighbours(self) -> dict[int, list[tuple[int, int]]]:
        adjacency: dict[int, list[tuple[int, int]]] = {
            index: [] for index in range(len(self.atoms))
        }
        for left, right, order in self.bonds:
            adjacency[left].append((right, order))
            adjacency[right].append((left, order))
        return adjacency

    def label(self) -> dict[str, int]:
        return {"atoms": len(self.atoms), "bonds": len(self.bonds)}


@dataclass(frozen=True)
class FingerprintSpec:
    """Bit width and radius of the hashed fingerprint.

    Ref: Sec. 2.6 and the response block of the configuration; the article prints
    neither width, so both are engineering defaults.
    """

    bits: int = 512
    radius: int = 2

    def index(self, code: int) -> int:
        return int(code % self.bits)


def _atom_code(symbol: str) -> int:
    return ATOM_CODES.get(symbol, 0)


def _fold(code: int, salt: int, modulus: int) -> int:
    digest = hashlib.blake2b(f"{code}:{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def morgan_like(molecule: Molecule, spec: FingerprintSpec) -> dict[int, int]:
    """Atom codes after ``spec.radius`` rounds of neighbour folding.

    The codes are returned per atom so the graph encoder and the fingerprint
    share one notion of an atom's environment.
    """
    adjacency = molecule.neighbours()
    codes = {index: _atom_code(symbol) for index, symbol in enumerate(molecule.atoms)}
    for round_index in range(spec.radius):
        updated = dict(codes)
        for index in range(len(molecule.atoms)):
            neighbours = sorted(
                (codes[other] * 10 + order, other) for other, order in adjacency.get(index, [])
            )
            combined = codes[index]
            for neighbour_code, other in neighbours:
                combined = _fold(combined * 31 + neighbour_code, other + round_index, 2**31)
            updated[index] = combined if neighbours else codes[index]
        codes = updated
    return codes


def hashed_fingerprint(molecule: Molecule, spec: FingerprintSpec) -> Array:
    """The folded bit vector of a molecule.

    The vector is real-valued and counts collisions rather than being binary, so
    two distinct substructures that fold to the same bit still leave a trace of
    how many of them there were.
    """
    codes = morgan_like(molecule, spec)
    vector = np.zeros(spec.bits, dtype=np.float64)
    for index, symbol in enumerate(molecule.atoms):
        vector[spec.index(codes[index])] += 1.0
        vector[spec.index(_atom_code(symbol))] += 0.5
    return np.asarray(vector, dtype=np.float64)


def fingerprint_batch(molecules: tuple[Molecule, ...], spec: FingerprintSpec) -> Array:
    """Stacked fingerprints, one row per molecule."""
    rows = [hashed_fingerprint(molecule, spec) for molecule in molecules]
    return np.asarray(np.stack(rows, axis=0), dtype=np.float64)
