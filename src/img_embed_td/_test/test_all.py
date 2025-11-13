import itertools
from pathlib import Path

import numpy as np
import pytest
import tracksdata as td

from img_embed_td import ImageEmbeddingConfig, ImageEmbeddingNodeAttrs
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
    "ndim,model_name",
    itertools.product(
        [2, 3],
        [
            "dinov3-vits16plus",
            "dinov3-convnext-tiny",
            "sam-base",
            "sam2-tiny",
        ],
    ),
)
def test_image_embedding_node_attrs(
    ndim: int,
    model_name: str,
) -> None:
    cfg = ImageEmbeddingConfig(
        model_name=model_name,
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
    model_name = "dinov3-vits16plus"

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
