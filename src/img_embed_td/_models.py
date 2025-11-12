from abc import ABC, abstractmethod
import torch
import numpy as np
import einops
import torch.nn.functional as F

from transformers import AutoConfig, AutoImageProcessor, AutoModel


class VisionModel(ABC):
    """Protocol for vision models."""

    def __init__(self, hub_name: str, model, processor):
        self._hub_name = hub_name
        self._model = model
        self._processor = processor

    @abstractmethod
    def extract_features(self, images_rgb: np.ndarray, image_size: int) -> torch.Tensor:
        """Extract features from RGB images."""
        ...

    @classmethod
    @abstractmethod
    def resolve_hub_name(cls, short_name: str) -> str:
        """Resolve a model by name."""
        ...

    @classmethod
    def model_builder(cls, short_name: str) -> "VisionModel":
        """Build a vision model."""
        hub_name = cls.resolve_hub_name(short_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = AutoConfig.from_pretrained(hub_name)
        processor = AutoImageProcessor.from_pretrained(hub_name)
        model = AutoModel.from_pretrained(hub_name, config=config)
        model = model.to(device).eval()

        return cls(hub_name, model, processor)
    
    def resize_images(self, images_rgb: np.ndarray, image_size: int) -> np.ndarray:
        """Resize images to the given size."""
        images_rgb = einops.rearrange(images_rgb, "... h w c -> ... c h w")
        images_rgb = F.interpolate(images_rgb, size=(image_size, image_size), mode="bilinear", align_corners=False)
        images_rgb = einops.rearrange(images_rgb, "... c h w -> ... h w c")
        return images_rgb.numpy()
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self._model.device


class DINOv3Model(VisionModel):
    """DINOv3 feature extractor."""

    def __init__(self, hub_name: str, model, processor):
        super().__init__(hub_name, model, processor)
        self._is_convnext = "convnext" in hub_name

    def extract_features(self, images_rgb: np.ndarray, image_size: int) -> torch.Tensor:
        """Extract features from RGB images."""

        inputs = self._processor(images=images_rgb, return_tensors="pt", size=image_size, do_center_crop=False)
        inputs = inputs["pixel_values"].to(self._model.device)

        with torch.no_grad():
            patches = self._model(inputs)

            if self._is_convnext:
                patches = patches.last_hidden_state[:, 1:]
            else:
                patches = patches.last_hidden_state[:, 5:]

        # reshape to spatial feature map
        patches = self._reshape_to_spatial(patches)

        return patches

    def _reshape_to_spatial(self, patches: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, D) patches to (B, D, H', W') spatial map."""
        batch_size, n_patches, embed_dim = patches.shape
        h_patches = w_patches = int(np.sqrt(n_patches))

        if h_patches * w_patches != n_patches:
            raise ValueError(
                f"Number of patches ({n_patches}) is not a perfect square. "
                f"Expected {h_patches}*{h_patches} = {h_patches * h_patches}."
            )

        # (B, N, D) -> (B, H', W', D) -> (B, D, H', W')
        spatial = patches.reshape(batch_size, h_patches, w_patches, embed_dim)
        return spatial.permute(0, 3, 1, 2)

    @classmethod
    def resolve_hub_name(cls, model_name: str) -> str:
        """Resolve a DINOv3 model by name."""
        _name_to_hub_name = {
            "dinov3-vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "dinov3-vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3-vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3-vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "dinov3-vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "dinov3-vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "dinov3-convnext-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            "dinov3-convnext-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            "dinov3-convnext-base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            "dinov3-convnext-large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        }

        if model_name not in _name_to_hub_name:
            raise ValueError(f"Unknown model name: {model_name}")

        return _name_to_hub_name[model_name]


class SAMModel(VisionModel):
    """SAM feature extractor."""

    def extract_features(self, images_rgb: np.ndarray, image_size: int) -> torch.Tensor:
        """Extract features from RGB images."""

        inputs = self._processor(images=images_rgb, return_tensors="pt", size=image_size)
        inputs = inputs["pixel_values"].to(self._model.device)

        with torch.no_grad():
            outputs = self._model.get_image_embeddings(inputs)

        return outputs

    @classmethod
    def resolve_hub_name(cls, model_name: str) -> str:
        """Resolve a SAM model by name."""
        _name_to_hub_name = {
            "sam-base": "facebook/sam-vit-base",
            "sam-large": "facebook/sam-vit-large",
            "sam-huge": "facebook/sam-vit-huge",
        }

        if model_name not in _name_to_hub_name:
            raise ValueError(f"Unknown model name: {model_name}")

        return _name_to_hub_name[model_name]
    
    def resize_images(self, images_rgb: np.ndarray, image_size: int) -> np.ndarray:
        """Resize images to the given size."""
        if image_size != 1024:
            raise ValueError(f"Image size must be 1024 for SAMv1 models, got {image_size}")
        return super().resize_images(images_rgb, image_size)


class SAM2Model(VisionModel):
    """SAM2 feature extractor."""

    def extract_features(self, images_rgb: np.ndarray, image_size: int) -> torch.Tensor:
        """Extract features from RGB images."""

        inputs = self._processor(images=images_rgb, return_tensors="pt", size=image_size)
        inputs = inputs["pixel_values"].to(self._model.device)

        with torch.no_grad():
            outputs = self._model.vision_encoder(inputs).fpn_hidden_states[0]

        return outputs

    @classmethod
    def resolve_hub_name(cls, model_name: str) -> str:
        """Resolve a SAM model by name."""
        _name_to_hub_name = {
            "sam2-tiny": "facebook/sam2.1-hiera-tiny",
            "sam2-small": "facebook/sam2.1-hiera-small",
            "sam2-base-plus": "facebook/sam2.1-hiera-base-plus",
            "sam2-large": "facebook/sam2.1-hiera-large",
        }

        if model_name not in _name_to_hub_name:
            raise ValueError(f"Unknown model name: {model_name}")

        return _name_to_hub_name[model_name]


MODEL_REGISTRY: dict[str, type[VisionModel]] = {
    "dinov3-vit7b16": DINOv3Model,
    "dinov3-vits16": DINOv3Model,
    "dinov3-vitb16": DINOv3Model,
    "dinov3-vitl16": DINOv3Model,
    "dinov3-vits16plus": DINOv3Model,
    "dinov3-vith16plus": DINOv3Model,
    "dinov3-convnext-tiny": DINOv3Model,
    "dinov3-convnext-small": DINOv3Model,
    "dinov3-convnext-base": DINOv3Model,
    "dinov3-convnext-large": DINOv3Model,
    "sam-base": SAMModel,
    "sam-large": SAMModel,
    "sam-huge": SAMModel,
    "sam2-tiny": SAM2Model,
    "sam2-small": SAM2Model,
    "sam2-base-plus": SAM2Model,
    "sam2-large": SAM2Model,
}

MODEL_NDIM: dict[str, int] = {
    "sam-base": 256,
    "sam-large": 256,
    "sam-huge": 256,
    "sam2-tiny": 768,
    "sam2-small": 768,
    "sam2-base-plus": 896,
    "sam2-large": 1152,
    "dinov3-vitb16": 768,
    "dinov3-vits16": 384,
    "dinov3-vitl16": 1024,
    "dinov3-vits16plus": 384,
    "dinov3-vit7b16": 4096,
    "dinov3-vith16plus": 1280,
}
