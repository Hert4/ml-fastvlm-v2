#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import json
from typing import Any

import torch.nn as nn

# Import GlobalPool2D for use in MCi class
from .mci import GlobalPool2D

# Import fastvithd directly - we'll use it instead of timm's create_model
# to avoid registration issues across different timm versions
from .mci import fastvithd


def load_model_config(
        model_name: str,
) -> Any:
    # Strip suffixes to model name
    model_name = "_".join(model_name.split("_")[0:2])

    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    if not os.path.exists(model_cfg_file):
        raise ValueError(f"Unsupported model name: {model_name}")
    model_cfg = json.load(open(model_cfg_file, "r"))

    return model_cfg


# Map model names to their factory functions
_MODEL_REGISTRY = {
    "fastvithd": fastvithd,
}


class MCi(nn.Module):
    """
    This class implements `MCi Models <https://arxiv.org/pdf/2311.17049.pdf>`_
    """

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        # Create model - use our own registry for fastvithd to avoid timm issues
        print(f"[MCi] Creating model: {model_name}")
        if model_name in _MODEL_REGISTRY:
            # Directly call the factory function
            print(f"[MCi] Using direct factory function for {model_name}")
            self.model = _MODEL_REGISTRY[model_name](**kwargs)
        else:
            # Fall back to timm's create_model for other models
            print(f"[MCi] Using timm create_model for {model_name}")
            from timm.models import create_model
            self.model = create_model(model_name, projection_dim=self.projection_dim)

        # Debug: verify model was created with parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MCi] Model created with {num_params:,} parameters")

        # Build out projection head.
        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head, projection_dim=self.projection_dim
                )

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model."""
        x = self.model(x, *args, **kwargs)
        return x

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Module) -> int:
        """Return the input feature dimension to the image classification head."""
        in_features = None
        if isinstance(image_classifier, nn.Sequential):
            # Classifier that uses nn.Sequential usually has global pooling and
            # multiple linear layers. Find the first linear layer and get its
            # in_features
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.in_features

        if in_features is None:
            raise NotImplementedError(
                f"Cannot get input feature dimension of {image_classifier}."
            )
        return in_features

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Module, projection_dim: int, *args, **kwargs
    ) -> nn.Module:
        in_features = MCi._get_in_feature_dimension(image_classifier)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier
