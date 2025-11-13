import itertools
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tracksdata as td

from img_embed_td import ImageEmbeddingConfig, ImageEmbeddingNodeAttrs
from img_embed_td._functional import _project_mask
from img_embed_td._models import MODEL_NDIM
from img_embed_td.cli import main


def _mock_data(ndim: int) -> tuple[td.graph.RustWorkXGraph, np.ndarray]:
    graph = td.graph.InMemoryGraph()

    positions = np.asarray(
        [
            [0, 5, 300, 150],  # t=0, z=5, y=10, x=20
            [0, 0, 0, 604],  # t=0, z=0, y=0, x=604
            [1, 6, 73, 250],  # t=1, z=6, y=15, x=25
            [1, 7, 210, 300],  # t=1, z=7, y=20, x=30
        ]
    )

    rng = np.random.default_rng(42)

    if ndim == 3:
        frames = rng.uniform(size=(2, 13, 418, 605))
    else:
        frames = rng.uniform(size=(2, 418, 605))
        positions = positions[:, [0, 2, 3]]  # removing z

    graph = td.graph.RustWorkXGraph.from_array(positions)

    mask_attrs = td.nodes.MaskDiskAttrs(
        radius=2,
        image_shape=frames.shape[1:],
        output_key=td.DEFAULT_ATTR_KEYS.MASK,
    )
    mask_attrs.add_node_attrs(graph)
    node_attrs = graph.node_attrs()

    graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.BBOX, np.zeros((ndim * 2,)))
    graph.update_node_attrs(
        attrs={
            td.DEFAULT_ATTR_KEYS.BBOX: [mask.bbox for mask in node_attrs[td.DEFAULT_ATTR_KEYS.MASK].to_list()],
        },
        node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
    )

    return graph, frames


@pytest.mark.parametrize(
    "ndim,model_name,mask_proj_mode",
    [
        *list(
            itertools.product(
                [2, 3],
                [
                    "dinov3-vits16plus",  # dino requires logging into HF
                    "dinov3-convnext-tiny",
                    "sam-base",
                    "sam2-tiny",
                ],
                ["none"],
            )
        ),
        (3, "dinov3-vits16plus", "max"),
    ],
)
def test_image_embedding_node_attrs(
    ndim: int,
    model_name: str,
    mask_proj_mode: Literal["none", "max"],
) -> None:
    cfg = ImageEmbeddingConfig(
        model_name=model_name,
        mask_proj_mode=mask_proj_mode,
    )
    embed_ops = ImageEmbeddingNodeAttrs(config=cfg)

    graph, frames = _mock_data(ndim)

    embed_ops.add_node_attrs(graph, frames=frames)

    assert cfg.model_name in graph.node_attr_keys


@pytest.mark.parametrize(
    "with_output_path",
    [
        True,
        False,
    ],
)
def test_cli(
    tmp_path: Path,
    with_output_path: bool,
) -> None:
    in_geff_path = tmp_path / "in_graph.geff"
    frames_path = tmp_path / "frames.npy"
    model_name = "sam2-tiny"

    cmd_and_args = [
        str(in_geff_path),
        str(frames_path),
        "-m",
        model_name,
    ]

    graph, frames = _mock_data(2)

    graph.to_geff(in_geff_path)
    np.save(frames_path, frames)

    if with_output_path:
        out_geff_path = tmp_path / "out_graph.geff"
        cmd_and_args.extend(["-o", str(out_geff_path)])
    else:
        out_geff_path = in_geff_path

    try:
        main(cmd_and_args)

    except SystemExit as e:
        assert e.code == 0, f"{cmd_and_args} failed with exit code {e.code}"

    out_graph, geff_metadata = td.graph.IndexedRXGraph.from_geff(out_geff_path)

    assert model_name in geff_metadata.node_props_metadata
    features = out_graph.node_attrs()[model_name].to_numpy()
    assert features.shape == (4, MODEL_NDIM[model_name])


def test_mask_projection() -> None:
    # Create a 3D mask with an asymmetric pattern
    mask_array = np.zeros((6, 15, 14), dtype=np.bool_)

    # Add asymmetric pattern across z-slices
    mask_array[2, 5:8, 3:6] = True  # pattern in z=2
    mask_array[3, 6:9, 4:7] = True  # shifted pattern in z=3
    mask_array[4, 4:6, 5:8] = True  # different pattern in z=4

    # Create a Mask object with bbox covering the whole volume
    bbox = np.array([2, 5, 7, 8, 20, 21])  # [z_min, y_min, x_min, z_max, y_max, x_max]
    mask = td.nodes.Mask(mask_array, bbox)

    # Test "none" projection mode at z=5 (3 + 2)
    mask_none = _project_mask(mask, mask_proj_mode="none", image_proj_window=3, z=5)

    # Should only get the z=3 slice
    expected_none = mask_array[3]
    assert mask_none.mask.shape == mask_array.shape[1:]
    assert np.array_equal(mask_none.mask, expected_none)
    assert np.array_equal(mask_none.bbox, np.array([5, 7, 20, 21]))  # [y_min, x_min, y_max, x_max]

    # Test "max" projection mode at z=5 (3 + 2) with window=3
    mask_max = _project_mask(mask, mask_proj_mode="max", image_proj_window=3, z=5)

    # Should get max projection of z=[2, 3, 4] (window=3 centered at z=3)
    expected_max = np.max(mask_array[2:5], axis=0)
    assert mask_max.mask.shape == mask_array.shape[1:]
    assert np.array_equal(mask_max.mask, expected_max)
    assert np.array_equal(mask_max.bbox, np.array([5, 7, 20, 21]))

    # Verify that the two modes give different results (asymmetric pattern)
    assert not np.array_equal(mask_none.mask, mask_max.mask)
