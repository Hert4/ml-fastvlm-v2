import math
import os
import re

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor


class MobileCLIP2VisionTower(nn.Module):
    """Vision tower wrapper for MobileCLIP2 checkpoints (e.g. apple/MobileCLIP2-S4)."""

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = getattr(args, 'input_image_size', None)

        self._cached_device = None
        self._cached_dtype = None

        image_size = self._parse_image_size(vision_tower)
        if self.input_image_size is not None:
            image_size = int(self.input_image_size)

        self.cfg_only = {
            "image_cfg": {
                "image_size": image_size,
                "patch_size": 16,
                "embed_dim": 3072,
            }
        }

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    @staticmethod
    def _parse_image_size(name: str) -> int:
        # Supports common variants like mobileclip_l_384.
        m = re.search(r"_(\d+)$", name)
        if m:
            return int(m.group(1))
        # MobileCLIP2 defaults
        return 384

    @staticmethod
    def _extract_features(output):
        # Dict outputs (common in vision libs)
        if isinstance(output, dict):
            for key in ("image_embeddings", "last_hidden_state", "x", "features"):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))

        # Tuple/list outputs
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Normalize to [B, N, C]
        if output.ndim == 4:
            b, c, h, w = output.shape
            output = output.reshape(b, c, h * w).transpose(1, 2)
        elif output.ndim == 2:
            output = output.unsqueeze(1)
        elif output.ndim != 3:
            raise ValueError(f"Unsupported MobileCLIP2 output shape: {tuple(output.shape)}")

        return output

    def _get_first_weight(self):
        try:
            return next(self.vision_tower.parameters())
        except Exception:
            return None

    def _infer_runtime_cfg(self, image_size: int):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            raw = self._forward_core(dummy)
            feats = self._extract_features(raw)

        hidden = int(feats.shape[-1])
        token_count = int(feats.shape[1])
        side = int(math.sqrt(token_count))
        if side * side == token_count and side > 0:
            patch_size = max(1, image_size // side)
        else:
            # Fallback for non-square tokenization.
            side = max(1, int(round(math.sqrt(max(1, token_count)))))
            patch_size = max(1, image_size // side)

        self.cfg_only = {
            "image_cfg": {
                "image_size": image_size,
                "patch_size": patch_size,
                "embed_dim": hidden,
            }
        }

    def _resolve_hf_hub_name(self):
        # timm hf-hub loader expects "hf-hub:<repo_id>"
        if self.vision_tower_name.startswith('hf-hub:'):
            return self.vision_tower_name
        return f"hf-hub:{self.vision_tower_name}"

    def _load_with_timm_hf_hub(self):
        import timm
        hub_name = self._resolve_hf_hub_name()
        return timm.create_model(hub_name, pretrained=True, num_classes=0)

    def _load_with_open_clip(self):
        # Fallback path requires open_clip_torch + ml-mobileclip
        from huggingface_hub import hf_hub_download, list_repo_files
        from open_clip.factory import create_model_and_transforms

        arch_name = os.path.basename(self.vision_tower_name)
        ckpt_file = os.environ.get("MOBILECLIP2_CKPT_FILE")
        if ckpt_file:
            ckpt_path = hf_hub_download(repo_id=self.vision_tower_name, filename=ckpt_file)
        else:
            repo_files = list_repo_files(self.vision_tower_name)
            candidates = [f for f in repo_files if f.lower().endswith((".pt", ".pth", ".bin"))]
            if not candidates:
                raise ValueError(
                    f"No .pt/.pth/.bin checkpoint found in repo {self.vision_tower_name}. "
                    "Set MOBILECLIP2_CKPT_FILE to a valid filename."
                )
            preferred = [f for f in candidates if 'mobileclip2' in f.lower()]
            pick = sorted(preferred or candidates)[0]
            ckpt_path = hf_hub_download(repo_id=self.vision_tower_name, filename=pick)

        model, _, _ = create_model_and_transforms(arch_name, pretrained=ckpt_path)
        try:
            from mobileclip.modules.common.mobileone import reparameterize_model
            model = reparameterize_model(model)
        except Exception:
            pass

        visual = getattr(model, 'visual', model)
        return getattr(visual, 'trunk', visual)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        image_size = self.cfg_only["image_cfg"]["image_size"]

        # Try timm hf-hub first (no extra deps beyond timm), then open_clip fallback.
        last_error = None
        try:
            self.vision_tower = self._load_with_timm_hf_hub()
            print(f"[MobileCLIP2] Loaded via timm hf-hub: {self.vision_tower_name}")
        except Exception as e:
            last_error = e
            try:
                self.vision_tower = self._load_with_open_clip()
                print(f"[MobileCLIP2] Loaded via open_clip fallback: {self.vision_tower_name}")
            except Exception as e2:
                raise RuntimeError(
                    "Failed to load MobileCLIP2 vision tower. "
                    "timm hf-hub path failed and open_clip fallback also failed.\n"
                    f"timm error: {last_error}\nopen_clip error: {e2}"
                )

        self.image_processor = CLIPImageProcessor(
            crop_size={"height": image_size, "width": image_size},
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            size={"shortest_edge": image_size},
        )

        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        # Infer hidden size/token geometry from runtime output.
        self._infer_runtime_cfg(image_size)

        self.is_loaded = True

    def _forward_core(self, images):
        if hasattr(self.vision_tower, 'forward_features'):
            return self.vision_tower.forward_features(images)
        return self.vision_tower(images)

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                raw = self._forward_core(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self._extract_features(raw).to(image.dtype)
                image_features.append(image_feature)
        else:
            raw = self._forward_core(images.to(device=self.device, dtype=self.dtype))
            image_features = self._extract_features(raw).to(images.dtype)

        return image_features

    def forward(self, images):
        if self.tune_vision_tower:
            return self.forward_images(images)
        else:
            with torch.no_grad():
                return self.forward_images(images)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self._cached_dtype is not None:
            return self._cached_dtype
        weight = self._get_first_weight()
        if weight is not None:
            self._cached_dtype = weight.dtype
            return weight.dtype
        return torch.float32

    @property
    def device(self):
        if self._cached_device is not None:
            return self._cached_device
        weight = self._get_first_weight()
        if weight is not None:
            self._cached_device = weight.device
            return weight.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config["image_cfg"]["embed_dim"]

    @property
    def num_patches_per_side(self):
        return self.config["image_cfg"]["image_size"] // max(1, self.config["image_cfg"]["patch_size"])

    @property
    def num_patches(self):
        side = self.num_patches_per_side
        return side * side
