from pathlib import Path
import click
import tracksdata as td


@click.command()
@click.argument()
@click.option(
    "--output-path",
    "-o"
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
    output_path: Path | None,
    overwrite: bool,
) -> None:

    graph, metadata = td.graph.IndexedRXGraph.from_geff(geff_path)

    # TODO: finish


if __name__ == "__main__":
    main()