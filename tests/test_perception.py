"""The perception front end: adapters, segmentation, tracking and embeddings."""

from __future__ import annotations

import numpy as np
import torch

from saber.cli.context import ReleaseContext
from saber.cohort.rendering import frame_bundle, instance_geometry, render_channel
from saber.perception.adapters import LoRAConfig, LoRAConv2d, count_adapter_parameters
from saber.perception.backbones import BackboneConfig, build_backbones
from saber.perception.head import FrameBundle
from saber.perception.segmenter import connected_instances, dice_coefficient
from saber.perception.tracker import TrackerState, associate
from saber.study import front_end_output


def test_rendered_geometry_is_deterministic(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    first = instance_geometry(specimen, context.panel.frame_size)
    second = instance_geometry(specimen, context.panel.frame_size)
    assert np.array_equal(first, second)


def test_rendered_frame_has_the_requested_shape(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    frame = render_channel(specimen, 0, "brightfield", context.panel.frame_size)
    assert frame.shape == (context.panel.frame_size, context.panel.frame_size)
    assert float(np.min(frame)) >= 0.0


def test_adapter_starts_as_the_identity_map() -> None:
    base = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1)
    adapted = LoRAConv2d(base, LoRAConfig(rank=3))
    inputs = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        assert torch.allclose(adapted(inputs), base(inputs))
        assert adapted.adapter_parameters() == 3 * 4 + 3 * 6


def test_adapter_is_the_only_trainable_part() -> None:
    vision, morphology = build_backbones(2, BackboneConfig(), adapter=LoRAConfig(rank=4))
    assert count_adapter_parameters(vision) > 0
    assert count_adapter_parameters(morphology) == 0


def test_connected_instances_labels_two_discs() -> None:
    image = np.zeros((16, 16), dtype=np.float64)
    image[2:6, 2:6] = 1.0
    image[10:14, 10:14] = 1.0
    labels = connected_instances(image, 0.5)
    assert int(labels.max()) == 2
    assert dice_coefficient(labels, labels) == 1.0


def test_tracker_keeps_an_identity_across_two_epochs() -> None:
    first = np.zeros((16, 16), dtype=np.int32)
    first[4:8, 4:8] = 1
    second = np.zeros((16, 16), dtype=np.int32)
    second[5:9, 5:9] = 1
    state = TrackerState(gate=4.0)
    completed = list(associate(state, first))
    completed.extend(associate(state, second))
    assert any(track.length == 2 for track in completed)


def test_front_end_emits_one_embedding_per_epoch(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    with torch.no_grad():
        output, stream = front_end_output(context, specimen)
    assert len(output.masks) == specimen.epochs
    assert int(stream.shape[0]) == specimen.epochs
    assert int(stream.shape[1]) == context.perception_spec.embedding_dim


def test_front_end_finds_instances_in_a_rendered_frame(context: ReleaseContext) -> None:
    frontend = context.build_frontend()
    frontend.eval()
    specimen = context.specimens()[0]
    bundle = FrameBundle(
        frames=frame_bundle(
            specimen, 0, context.perception_spec.channels, context.panel.frame_size
        ),
        channel_names=context.perception_spec.channels,
        specimen=specimen.identity,
        epoch=0,
    )
    with torch.no_grad():
        output = frontend(bundle, context.perception_spec.tracker_gate)
    assert int(np.max(output.masks[0])) >= 1


def test_channel_selection_restricts_the_bundle(context: ReleaseContext) -> None:
    specimen = context.specimens()[0]
    channels = ("brightfield", "phase_contrast", "fluorescence")
    bundle = FrameBundle(
        frames=frame_bundle(specimen, 0, channels, context.panel.frame_size),
        channel_names=channels,
        specimen=specimen.identity,
        epoch=0,
    )
    subset = bundle.select(("phase_contrast",))
    assert subset.channel_names == ("phase_contrast",)
    assert int(subset.frames.shape[1]) == 1
