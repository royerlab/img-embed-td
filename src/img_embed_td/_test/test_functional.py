import numpy as np
import pytest
import tracksdata as td

from img_embed_td import ImageEmbeddingConfig, ImageEmbeddingNodeAttrs


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

    return graph, frames


@pytest.mark.parametrize("ndim", [2, 3])
def test_image_embedding_node_attrs(
    ndim: int,
) -> None:
    cfg = ImageEmbeddingConfig(
        model_name="dinov3-vits16plus",
    )
    embed_ops = ImageEmbeddingNodeAttrs(config=cfg)

    graph, frames = _mock_data(ndim)

    embed_ops.add_node_attrs(graph, frames=frames)

    assert cfg.model_name in graph.node_attr_keys
