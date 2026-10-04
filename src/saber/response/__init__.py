"""Microenvironment-conditioned response prediction.

Ref: Sec. 2.6 -- ``f_phi`` predicts the expected dose-response summary
``y_hat = f_phi(b_t, u)`` from the inferred belief and a drug representation ``u``
built from "a molecular graph and a hashed fingerprint"; the unconditioned model
"follows the same structure and the same training dataset ... except for the fact
that instead of replacing the believed coordinates, the believed coordinates are
omitted completely". Sec. 3.6 reports the directed cross-corpus transfer of both.
"""

from saber.response.fingerprint import (
    FingerprintSpec,
    Molecule,
    hashed_fingerprint,
    morgan_like,
)
from saber.response.head import ResponseHead, ResponsePrediction, dose_response_summary
from saber.response.molecular import GraphEncoder, atom_features, bond_index
from saber.response.transfer import (
    CorpusPrediction,
    TransferMatrix,
    directed_transfer,
    retention_table,
)

__all__ = [
    "CorpusPrediction",
    "FingerprintSpec",
    "GraphEncoder",
    "Molecule",
    "ResponseHead",
    "ResponsePrediction",
    "TransferMatrix",
    "atom_features",
    "bond_index",
    "directed_transfer",
    "dose_response_summary",
    "hashed_fingerprint",
    "morgan_like",
    "retention_table",
]
