import numpy as np
from cmap import Colormap
from numpy.typing import ArrayLike


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
        raise ValueError(f"Expected 3D array, got shape {image.shape}")

    half = window // 2
    start = max(0, z - half)
    end = min(image.shape[0], start + window)
    return np.asarray(image[start:end]).max(axis=0)


class ColorMapper:
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

    def __init__(self, cmap_name: str) -> None:
        if cmap_name == "none":
            self._cmap = None
        else:
            self._cmap = Colormap(cmap_name)

    def __call__(self, image: ArrayLike) -> np.ndarray:
        if self._cmap is None:
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Colormap 'none' expects (H, W, 3) array per slice, got shape {image.shape}")

            min_val = image.min()
            max_val = image.max()
            if min_val < 0 or max_val > 255 or max_val <= 1.1:  # 1.1 for a tolerance in normalization
                raise ValueError(
                    "For colormap 'none' the image must be preprocessed to be in [0, 255] range"
                    f"Found values in [{min_val}, {max_val}]"
                )
            return image
        return (self._cmap(image) * 255)[..., :3].astype(np.uint8)
