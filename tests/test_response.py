"""The response head, the fingerprint, the graph encoder and the transfer table."""

from __future__ import annotations

import numpy as np
import torch

from saber.belief.pipeline import initial_belief
from saber.cli.context import ReleaseContext
from saber.response.fingerprint import FingerprintSpec, Molecule, hashed_fingerprint, morgan_like
from saber.response.head import DOSE_RANGE_M, hill_curve, hill_fit, hill_r_squared, summarise_curve
from saber.response.molecular import atom_features, bond_index
from saber.response.transfer import (
    CorpusArm,
    directed_transfer,
    importance_weights,
    retention_table,
)

MOLECULE = Molecule(atoms=("C", "C", "N", "O"), bonds=((0, 1, 1), (1, 2, 1), (2, 3, 2)))
OTHER = Molecule(atoms=("C", "C", "S", "O"), bonds=((0, 1, 1), (1, 2, 1), (2, 3, 2)))


def test_fingerprint_is_deterministic_and_sized() -> None:
    spec = FingerprintSpec(bits=128, radius=2)
    assert np.array_equal(hashed_fingerprint(MOLECULE, spec), hashed_fingerprint(MOLECULE, spec))
    assert hashed_fingerprint(MOLECULE, spec).size == 128


def test_fingerprint_separates_a_heteroatom() -> None:
    spec = FingerprintSpec(bits=256, radius=2)
    assert not np.array_equal(hashed_fingerprint(MOLECULE, spec), hashed_fingerprint(OTHER, spec))


def test_fingerprint_radius_changes_the_environment_codes() -> None:
    shallow = morgan_like(MOLECULE, FingerprintSpec(bits=256, radius=1))
    deep = morgan_like(MOLECULE, FingerprintSpec(bits=256, radius=3))
    assert shallow != deep


def test_atom_features_and_bonds_have_the_graph_shape() -> None:
    features = atom_features(MOLECULE)
    edge_index, edge_features, orders = bond_index(MOLECULE)
    assert features.shape[0] == len(MOLECULE.atoms)
    assert edge_index.shape[1] == 2 * len(MOLECULE.bonds)
    assert edge_features.shape[0] == 2 * len(MOLECULE.bonds)
    assert orders.shape[0] == 2 * len(MOLECULE.bonds)


def test_hill_fit_recovers_a_known_curve() -> None:
    doses = np.logspace(np.log10(DOSE_RANGE_M[0]), np.log10(DOSE_RANGE_M[1]), 9)
    truth = (1.0, 0.1, -6.0, 1.0)
    response = hill_curve(doses, *truth)
    fitted = hill_fit(doses, response)
    assert hill_r_squared(doses, response, fitted) >= 0.999
    assert abs(fitted[2] - truth[2]) <= 0.05


def test_summary_has_the_configured_number_of_points() -> None:
    summary = summarise_curve((1.0, 0.1, -6.0, 1.0), 5)
    assert summary.shape == (5,)


def test_head_shapes_and_conditioning(context: ReleaseContext) -> None:
    head = context.build_response_head()
    spec = FingerprintSpec(bits=context.response_spec.fingerprint_bits, radius=1)
    fingerprint = hashed_fingerprint(MOLECULE, spec)
    prediction = head(initial_belief(context.prior), fingerprint, MOLECULE)
    assert prediction.summary.shape == (context.response_spec.dose_summary_points,)
    assert prediction.conditioned


def test_unconditioned_head_omits_the_coordinates(context: ReleaseContext) -> None:
    spec = context.response_spec
    from saber.response.head import ResponseHead

    head = ResponseHead(
        belief_dim=4,
        graph_dim=spec.hidden_dim // 2,
        fingerprint_bits=spec.fingerprint_bits,
        hidden_dim=spec.hidden_dim,
        points=spec.dose_summary_points,
        conditioned=False,
    )
    assert not head.conditioned


def test_head_is_differentiable(context: ReleaseContext) -> None:
    head = context.build_response_head()
    spec = FingerprintSpec(bits=context.response_spec.fingerprint_bits, radius=1)
    fingerprint = hashed_fingerprint(MOLECULE, spec)
    prediction = head.predict_tensor(initial_belief(context.prior), fingerprint, MOLECULE)
    loss = torch.sum(prediction**2)
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) >= 0.0
        for parameter in head.parameters()
    )


def test_directed_transfer_covers_every_pair(context: ReleaseContext) -> None:
    belief = initial_belief(context.prior)
    spec = FingerprintSpec(
        bits=context.response_spec.fingerprint_bits,
        radius=context.response_spec.fingerprint_radius,
    )
    fingerprint = hashed_fingerprint(MOLECULE, spec)
    observed = np.asarray([[0.9, 0.7, 0.4, 0.2, 0.1]], dtype=np.float64)
    arms = tuple(
        CorpusArm(
            name=name,
            beliefs=(belief,),
            fingerprints=fingerprint[None, :],
            molecules=(MOLECULE,),
            observed=observed,
            baseline=np.full_like(observed, float(observed.mean())),
            weights=np.ones(1, dtype=np.float64),
        )
        for name in ("A", "B")
    )
    matrix = directed_transfer(
        lambda: context.build_response_head(), arms, arms, epochs=3, floor=400.0
    )
    assert len(matrix.predictions) == 4
    assert matrix.within_corpus_range()[0] <= matrix.within_corpus_range()[1]
    assert len(retention_table(matrix)) == 4


def test_importance_weights_are_a_probability_ratio() -> None:
    weights = importance_weights(np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5]))
    assert abs(float(weights.sum()) - 1.0) <= 1e-12


def test_importance_weights_zero_a_hopeless_record() -> None:
    weights = importance_weights(np.asarray([0.5, 0.5]), np.asarray([0.0, 1.0]))
    assert float(weights[0]) == 0.0
