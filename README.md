# Image Embedding For Tracking

A toolbox for image embedding feature extraction for [tracksdata](https://github.com/royerlab/tracksdata) graphs.

## Installation

```bash
pip install git+https://github.com/royerlab/img-embed-td.git
```

## Quick Start

[tracksdata](https://github.com/royerlab/tracksdata) compatibility through python

```python
import torch
import tracksdata as td
from img_embed_td import ImageEmbeddingConfig, ImageEmbeddingNodeAttrs

graph = ...

config = ImageEmbeddingConfig(
    model_name="dinov3-vits16plus",
)

embed_ops = ImageEmbeddingNodeAttrs(config=config)
embed_ops.add_node_attrs(graph, frames=frames)
```

[geff](https://github.com/live-image-tracking-tools/geff) CLI compatibility

```bash
# in-place addition of embedding attributes to the graph
img-embed-td your_graph.geff directory_with_images/*.tif -m dinov3-vits16plus
```
