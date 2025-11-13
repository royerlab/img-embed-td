import warnings
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td
from cmap import Colormap
from numpy.typing import ArrayLike
from pydantic import BaseModel, model_validator
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from tracksdata.graph import BaseGraph
from tracksdata.nodes._generic_nodes import BaseNodeAttrsOperator
from tracksdata.nodes._mask import Mask

from img_embed_td._models import MODEL_NDIM, MODEL_REGISTRY
from img_embed_td._transforms import apply_colormap, select_slice_with_max_proj


def _mixed_collate(
    batch: list[tuple[pl.DataFrame, torch.Tensor]],
) -> tuple[list[pl.DataFrame], torch.Tensor]:
    dfs, tensors = zip(*list(batch), strict=False)
    return dfs, torch.stack(tensors)


class ImageEmbeddingConfig(BaseModel):
    """
    Configuration for image embedding.
    """

    model_name: str
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bilinear"
    colormap: Literal["gray", "jet", "viridis", "plasma", "inferno", "magma", "cividis", "turbo", "none"] = "gray"
    volume_mode: Literal["none", "slice", "k_max"] = "none"

    k_max_window: int = 3
    batch_size: int = 8
    norm_vectors: bool = True
    # obs: SAM models will ignore this by architecture, images are resized to 1024
    model_image_size: int = 512

    model_config = {"arbitrary_types_allowed": False}

    @model_validator(mode="after")
    def _validate_sam_image_size(self) -> "ImageEmbeddingConfig":
        if self.model_name.lower().startswith("sam-"):
            if self.model_image_size != 1024:
                warnings.warn(
                    "SAM models will ignore `model_image_size` by architecture, images are resized to 1024",
                    stacklevel=2,
                )
            self.model_image_size = 1024
        return self


class SlicesDataset(IterableDataset):
    def __init__(self, graph: BaseGraph, config: ImageEmbeddingConfig, frames: ArrayLike) -> None:
        self._graph = graph
        self._config = config
        self._frames = frames
        self._cmap = Colormap(config.colormap)

    def __iter__(self) -> Iterator[tuple[pl.DataFrame, torch.Tensor]]:
        attr_keys = [td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T, td.DEFAULT_ATTR_KEYS.MASK]

        is_2d = True
        if "z" in self._graph.node_attr_keys:
            attr_keys.append("z")
            is_2d = False

        node_attrs = self._graph.node_attrs(attr_keys=attr_keys)

        for (t,), t_group in node_attrs.group_by(td.DEFAULT_ATTR_KEYS.T):
            frame = np.asarray(self._frames[t])

            if is_2d:
                frame = apply_colormap(frame, self._cmap)
                yield t_group, torch.from_numpy(frame)

            else:
                for (z,), group_z in t_group.group_by("z"):
                    max_proj = select_slice_with_max_proj(frame, z, self._config.k_max_window)
                    max_proj = apply_colormap(max_proj, self._cmap)
                    yield group_z, torch.from_numpy(max_proj)


@contextmanager
def _amp_context(device_type: str) -> Generator[None, None, None]:
    if device_type == "cuda":
        with torch.amp.autocast(device_type=device_type):
            yield
    else:
        yield


class ImageEmbeddingNodeAttrs(BaseNodeAttrsOperator):
    def __init__(
        self,
        config: ImageEmbeddingConfig,
    ) -> None:
        self.config = config
        self.output_key = config.model_name

        model_cls = MODEL_REGISTRY[self.config.model_name]
        self.model = model_cls.model_builder(self.config.model_name)

    def _init_node_attrs(self, graph: BaseGraph) -> None:
        if self.output_key not in graph.node_attr_keys:
            ndim = MODEL_NDIM[self.config.model_name]
            graph.add_node_attr_key(self.output_key, default_value=np.zeros((ndim,), dtype=np.float32))

    def _node_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        **kwargs: Any,
    ) -> tuple[list[int], dict[str, list[Any]]]:
        raise NotImplementedError("`_node_attrs_per_time` is not available for `ImageEmbeddingNodeAttrs`")

    def add_node_attrs(
        self,
        graph: BaseGraph,
        *,
        frames: ArrayLike,
    ) -> None:
        self._init_node_attrs(graph)

        is_2d = "z" not in graph.node_attr_keys

        dataset = SlicesDataset(graph, self.config, frames)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=_mixed_collate)

        with torch.no_grad(), _amp_context(self.model.device.type):
            for batch in tqdm(dataloader, desc="Embedding images"):
                batch_df: list[pl.DataFrame] = batch[0]
                images: torch.Tensor = batch[1]

                orig_shape = images.shape
                images = self.model.resize_images(images, self.config.model_image_size)
                features = self.model.extract_features(images, self.config.model_image_size)
                features = F.interpolate(
                    features,
                    size=orig_shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                )
                # returning to original shape
                features = features.permute(0, 2, 3, 1).float().cpu().numpy()

                node_ids = []
                node_features = []
                for df, slice_features in zip(batch_df, features, strict=False):
                    for row in df.rows(named=True):
                        mask: Mask = row[td.DEFAULT_ATTR_KEYS.MASK]

                        if not is_2d:
                            z_step = row["z"] - mask.bbox[0]
                            slice_mask = mask.mask[z_step]
                            # converting to 2D mask
                            mask = Mask(slice_mask, mask.bbox[[1, 2, 4, 5]])

                        mask_features = mask.crop(slice_features)[mask.mask]
                        if self.config.norm_vectors:
                            mask_features = mask_features / np.linalg.norm(mask_features, axis=1, keepdims=True).clip(
                                min=1e-6
                            )
                        node_features.append(mask_features.mean(axis=0))
                        node_ids.append(row[td.DEFAULT_ATTR_KEYS.NODE_ID])

                graph.update_node_attrs(
                    node_ids=node_ids,
                    attrs={
                        self.output_key: node_features,
                    },
                )
