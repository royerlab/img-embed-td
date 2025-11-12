
import numpy as np
from numpy.typing import ArrayLike
import einops
from cmap import Colormap


def select_slice_with_max_proj(
    image: ArrayLike,
    z: int,
    window: int,
) -> np.ndarray:
    """
    Select the slice with the maximum projection along the Z axis.

    Parameters
    ----------
    image : ArrayLike
        Image to select slice from.
    z : int
        Z-coordinate of the slice to select.
    window : int
        Window size to use for the max projection.

    Returns
    -------
    np.ndarray
        Selected slice.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got shape {image.shape}"
        )

    half = window // 2
    start = max(0, z - half)
    end = min(image.shape[0], start + window)
    return np.asarray(image[start:end]).max(axis=0)


def apply_colormap(
    image: ArrayLike,
    cmap: Colormap,
) -> np.ndarray:
    """
    Apply colormap to image.

    Parameters
    ----------
    image : ArrayLike
        Image to apply colormap to.
    cmap : Colormap
        Colormap to apply to the image.

    Returns
    -------
    np.ndarray
        RGB image.
    """
    rgb = (cmap(image) * 255)[..., :3].astype(np.uint8)
    return rgb
