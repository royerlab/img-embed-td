from pathlib import Path

import click
import dask.array as da
import imageio
import tracksdata as td
import zarr
from dask.array.image import imread as dask_imread

from img_embed_td._functional import (
    MODEL_REGISTRY,
    ImageEmbeddingConfig,
    ImageEmbeddingNodeAttrs,
)


@click.command()
@click.argument("geff_path", type=click.Path(path_type=Path))
@click.argument("frames_path", type=click.Path(path_type=Path))
@click.option(
    "--model-name",
    "-m",
    type=click.Choice(MODEL_REGISTRY.keys()),
    help="Model name to use for embedding.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    help="Path to geff, if not provided, new attribute will be added to existing geff.",
)
@click.option(
    "--overwrite",
    "-ow",
    is_flag=True,
    default=False,
    type=bool,
    help="Overwrite embedding or geff if it already exists.",
)
def main(
    geff_path: Path,
    frames_path: Path,
    model_name: str,
    output_path: Path | None,
    overwrite: bool,
) -> None:
    graph, metadata = td.graph.IndexedRXGraph.from_geff(geff_path)

    # TODO: overwrite checking

    if "*" in str(frames_path):
        frames = dask_imread(frames_path)
    elif frames_path.is_dir():
        frames = da.from_zarr(zarr.open(frames_path))
    else:
        frames = imageio.imread(frames_path)

    cfg = ImageEmbeddingConfig(
        model_name=model_name,
    )

    embed_ops = ImageEmbeddingNodeAttrs(config=cfg)
    embed_ops.add_node_attrs(graph, frames=frames)

    # TODO: saving


if __name__ == "__main__":
    main()
