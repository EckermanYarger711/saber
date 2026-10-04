"""The release cohort, its records, the registry and the frames."""

from __future__ import annotations

import numpy as np

from saber.cli.context import ReleaseContext
from saber.cohort.registry import (
    CORPUS_ROLES,
    RESOURCES,
    CorpusCatalogue,
    registry_rows,
    resources_without_licence_position,
)
from saber.cohort.release_cohort import read_cohort, write_cohort
from saber.cohort.rendering import frame_bundle, render_channel
from saber.cohort.schema import CohortRecord, SpecimenRecord


def test_cohort_counts_match_the_panel(context: ReleaseContext) -> None:
    catalogue = CorpusCatalogue.from_panel(context.panel)
    assert catalogue.total == context.panel.specimens


def test_slice_covers_every_corpus_and_stratum(context: ReleaseContext) -> None:
    assert len(context.cohort.corpora()) == 5
    assert len(context.cohort.strata()) == 5


def test_specimen_records_have_the_session_shape(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    assert specimen.axes.shape == (context.twin_spec.epochs, 4)
    assert specimen.observations.shape == specimen.axes.shape
    assert specimen.dose_response.shape == (context.response_spec.dose_summary_points,)
    assert specimen.epochs == context.twin_spec.epochs


def test_cohort_round_trips_through_disk(context: ReleaseContext, tmp_path: object) -> None:
    write_cohort(context.cohort, tmp_path)
    reloaded = read_cohort(tmp_path)
    assert reloaded.size() == context.cohort.size()
    for left, right in zip(context.cohort.specimens, reloaded.specimens):
        assert np.array_equal(left.axes, right.axes)
        assert left.identity == right.identity
        assert left.stratum == right.stratum


def test_registry_lists_each_resource_once() -> None:
    names = [resource.name for resource in RESOURCES]
    assert len(names) == len(set(names))
    assert all(resource.licence and resource.accessed for resource in RESOURCES)


def test_registry_rows_are_serialisable() -> None:
    rows = registry_rows()
    assert len(rows) == len(RESOURCES)
    assert all(set(row) == {"name", "version", "licence", "accessed", "role"} for row in rows)


def test_registry_records_the_no_licence_position() -> None:
    holders = resources_without_licence_position()
    assert 0 < len(holders) < len(RESOURCES)


def test_every_corpus_has_a_role() -> None:
    assert set(CORPUS_ROLES) == set("ABCDE")


def test_rendered_bundle_has_the_channel_count(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    channels = context.perception_spec.channels
    bundle = frame_bundle(specimen, 0, channels, context.panel.frame_size)
    assert int(bundle.shape[1]) == len(channels)


def test_frames_differ_across_channels(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    bright = render_channel(specimen, 0, "brightfield", context.panel.frame_size)
    fluorescence = render_channel(specimen, 0, "fluorescence", context.panel.frame_size)
    assert not np.array_equal(bright, fluorescence)


def test_cohort_record_reports_its_summary() -> None:
    record = CohortRecord(
        specimens=(
            SpecimenRecord(
                identity="A-0001",
                corpus="A",
                stratum="hypoxic",
                transition_rate=0.41,
                axes=np.zeros((2, 4)),
                observations=np.zeros((2, 4)),
                dose_response=np.zeros(5),
            ),
        ),
        epoch_hours=6.0,
        dose_points=5,
    )
    assert record.size() == 1
    assert record.summary()["corpora"] == {"A": 1}
