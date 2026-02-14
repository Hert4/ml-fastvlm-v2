#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor
import llava.model.multimodal_encoder.mobileclip as mobileclip


class MobileCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = int(vision_tower.split("_")[-1])

        # Initialize cached device/dtype (will be set when model is loaded/moved)
        self._cached_device = None
        self._cached_dtype = None

        # Delay load is disabled for now
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            model_cfg = mobileclip.load_model_config(self.vision_tower_name)
            self.cfg_only = model_cfg

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Load model config
        model_cfg = mobileclip.load_model_config(self.vision_tower_name)

        # Override default image resolution
        model_cfg["image_cfg"]["image_size"] = self.input_image_size

        self.cfg_only = model_cfg

        # Build HF CLIPImageProcessor with MobileCLIP parameters
        self.image_processor = CLIPImageProcessor(crop_size={"height": model_cfg["image_cfg"]["image_size"],
                                                             "width": model_cfg["image_cfg"]["image_size"]},
                                                  image_mean=[0.0, 0.0, 0.0],
                                                  image_std=[1.0, 1.0, 1.0],
                                                  size={"shortest_edge": model_cfg["image_cfg"]["image_size"]})

        # Instantiate the image encoder
        self.vision_tower = mobileclip.MCi(model_name=model_cfg["image_cfg"]["model_name"],
                                           projection_dim=model_cfg["embed_dim"])

        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # Features from penultimate layer
        image_features = image_forward_outs["image_embeddings"]

        # Reshape 4D tensor to 3D
        B, C, H, W = image_features.shape
        image_features = image_features.reshape(B, C, H*W)
        image_features = image_features.transpose(1, 2)
        return image_features

    def forward(self, images):
        if self.tune_vision_tower:
            return self.forward_images(images)
        else:
            with torch.no_grad():
                return self.forward_images(images)

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), return_image_embeddings=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), return_image_embeddings=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    def _get_first_weight(self):
        """Get the first weight tensor from the vision tower, bypassing parameter iteration issues."""
        # Try to get a weight directly from a known layer
        try:
            # Access the inner model's first conv weight directly
            if hasattr(self.vision_tower, 'model'):
                model = self.vision_tower.model
                # Try patch_embed first (convolutional stem)
                if hasattr(model, 'patch_embed') and len(model.patch_embed) > 0:
                    first_module = model.patch_embed[0]
                    if hasattr(first_module, 'reparam_conv') and first_module.reparam_conv is not None:
                        return first_module.reparam_conv.weight
                    if hasattr(first_module, 'rbr_conv') and first_module.rbr_conv is not None:
                        return first_module.rbr_conv[0].conv.weight
                # Try conv_exp
                if hasattr(model, 'conv_exp'):
                    if hasattr(model.conv_exp, 'reparam_conv') and model.conv_exp.reparam_conv is not None:
                        return model.conv_exp.reparam_conv.weight
        except Exception as e:
            print(f"[MobileCLIPVisionTower] Error getting weight directly: {e}")

        # Fallback to parameter iteration
        try:
            return next(self.vision_tower.parameters())
        except StopIteration:
            return None

    @property
    def dtype(self):
        # Use cached dtype if available
        if self._cached_dtype is not None:
            return self._cached_dtype

        weight = self._get_first_weight()
        if weight is not None:
            self._cached_dtype = weight.dtype
            return weight.dtype

        # Final fallback
        import torch
        return torch.float32

    @property
    def device(self):
        # Use cached device if available
        if self._cached_device is not None:
            return self._cached_device

        weight = self._get_first_weight()
        if weight is not None:
            self._cached_device = weight.device
            return weight.device

        # Final fallback
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config["image_cfg"]["embed_dim"]

    @property
    def num_patches_per_side(self):
        return self.config["image_cfg"]["image_size"] // self.config["image_cfg"]["patch_size"]

    @property
    def num_patches(self):
        return (self.config["image_cfg"]["image_size"] // self.config["image_cfg"]["patch_size"]) ** 2
