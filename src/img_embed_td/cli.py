import shutil
from pathlib import Path

import click
import dask.array as da
import imageio
import numpy as np
import tracksdata as td
import zarr
from dask.array.image import imread as dask_imread
from geff.core_io._base_write import write_props_arrays

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

    if output_path is None:
        if model_name in metadata.node_props_metadata and not overwrite:
            raise ValueError(
                f"Property '{model_name}' already exists in {geff_path}, use `--overwrite` to overwrite it"
            )
        output_path = geff_path

    else:
        if output_path.exists():
            if not overwrite:
                raise ValueError(f"Output path {output_path} already exists, use `--overwrite` to overwrite it")
        else:
            shutil.copytree(geff_path, output_path)

    if "*" in str(frames_path):
        frames = dask_imread(frames_path)
    elif frames_path.name.endswith(".npy"):
        frames = np.load(frames_path)
    elif frames_path.is_dir():
        frames = da.from_zarr(zarr.open(frames_path))
    else:
        frames = imageio.imread(frames_path)

    cfg = ImageEmbeddingConfig(
        model_name=model_name,
    )

    embed_ops = ImageEmbeddingNodeAttrs(config=cfg)
    embed_ops.add_node_attrs(graph, frames=frames)

    zarr_version = 3 if (output_path / "zarr.json").exists() else 2

    # only writing new values
    prop_metadata = write_props_arrays(
        output_path,
        group="nodes",
        props={
            model_name: {
                "values": graph.node_attrs(attr_keys=model_name)[model_name].to_numpy(),
                "missing": None,
            },
        },
        zarr_format=zarr_version,
    )[0]
    prop_metadata.description = f"Image embedding extracted with {model_name} model"
    metadata.node_props_metadata[model_name] = prop_metadata
    metadata.write(output_path)


if __name__ == "__main__":
    main()
