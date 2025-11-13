import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sam2-tiny-only",
        action="store_true",
        default=False,
        help="Run only test_image_embedding_node_attrs with sam2-tiny model",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "sam2_tiny_only: mark test to run only with --sam2-tiny-only flag")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--sam2-tiny-only"):
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        # Check if this is the test_image_embedding_node_attrs test
        if item.originalname == "test_image_embedding_node_attrs":
            # Check if this is the sam2-tiny variant
            # The parametrize creates test IDs like: test_image_embedding_node_attrs[2-sam2-tiny]
            if "sam2-tiny" in item.nodeid:
                selected.append(item)
            else:
                deselected.append(item)
        else:
            # Skip all other tests
            deselected.append(item)

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
