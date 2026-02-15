# LLaVA Qwen3 Model for HuggingFace
# Upload this file to your HuggingFace repo for trust_remote_code=True

from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Qwen3 imports
try:
    from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    Qwen3Config = Qwen2Config
    Qwen3Model = Qwen2Model
    Qwen3ForCausalLM = Qwen2ForCausalLM

from .configuration_llava_qwen import LlavaQwen3Config

# Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


# ============== Vision Projector ==============

def build_vision_projector(config, delay_load=False, **kwargs):
    """Build vision projector based on config."""
    projector_type = getattr(config, "mm_projector_type", "linear")
    mm_hidden_size = getattr(config, "mm_hidden_size", 1024)
    hidden_size = config.hidden_size

    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)

    if projector_type == "mlp2x_gelu":
        return nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    if projector_type == "identity":
        return nn.Identity()

    raise ValueError(f"Unknown projector type: {projector_type}")


# ============== MobileCLIP Vision Tower ==============

class MobileCLIPVisionTower(nn.Module):
    """MobileCLIP Vision Tower for FastVLM."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_loaded = False
        self.image_processor = None
        self.vision_tower = None

        # Get vision tower name from config
        self.vision_tower_name = getattr(config, "mm_vision_tower", "mobileclip_l_384")

        # Parse image size from name (e.g., mobileclip_l_384 -> 384)
        try:
            self.image_size = int(self.vision_tower_name.split("_")[-1])
        except:
            self.image_size = 384

        # Hidden size for mobileclip_l variants
        self.hidden_size = getattr(config, "mm_hidden_size", 3072)

    def load_model(self, **kwargs):
        """Load the MobileCLIP model."""
        if self.is_loaded:
            return

        try:
            import timm
            from transformers import CLIPImageProcessor

            # Load FastViTHD model from timm
            # MobileCLIP uses fastvit_mci architecture
            self.vision_tower = timm.create_model(
                "fastvit_mci2.apple_mclip",
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            self.vision_tower.eval()

            # Setup image processor
            self.image_processor = CLIPImageProcessor(
                size={"shortest_edge": self.image_size},
                crop_size={"height": self.image_size, "width": self.image_size},
                do_center_crop=True,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )

            self.is_loaded = True
            print(f"MobileCLIP vision tower loaded: {self.vision_tower_name}")

        except Exception as e:
            print(f"Warning: Could not load MobileCLIP: {e}")
            print("Falling back to simple CNN encoder...")
            self._create_fallback_encoder()
            self.is_loaded = True

    def _create_fallback_encoder(self):
        """Create a simple fallback encoder if MobileCLIP fails to load."""
        from transformers import CLIPImageProcessor

        # Simple CNN that outputs correct hidden size
        self.vision_tower = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((24, 24)),  # 576 patches
            nn.Flatten(2),
            nn.Linear(256, self.hidden_size),
        )

        self.image_processor = CLIPImageProcessor(
            size={"shortest_edge": self.image_size},
            crop_size={"height": self.image_size, "width": self.image_size},
        )

    def forward(self, images):
        """Forward pass through vision tower."""
        if not self.is_loaded:
            self.load_model()

        if self.vision_tower is None:
            raise RuntimeError("Vision tower not loaded")

        # Get features
        with torch.no_grad():
            if hasattr(self.vision_tower, 'forward_features'):
                # timm model
                features = self.vision_tower.forward_features(images)
            else:
                # fallback
                features = self.vision_tower(images)

        # Reshape to (batch, num_patches, hidden_size)
        if features.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
        elif features.dim() == 2:
            # (B, C) -> (B, 1, C)
            features = features.unsqueeze(1)

        return features

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.vision_tower is not None:
            self.vision_tower = self.vision_tower.to(*args, **kwargs)
        return self


def build_vision_tower(config, delay_load=False, **kwargs):
    """Build vision tower."""
    return MobileCLIPVisionTower(config)


# ============== LLaVA Meta Classes ==============

class LlavaMetaModel:
    """Mixin for LLaVA model with vision capabilities."""

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower") and config.mm_vision_tower is not None:
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class LlavaMetaForCausalLM(ABC):
    """Abstract base for LLaVA causal LM."""

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            vision_tower = vision_tower.to(device=images.device, dtype=images.dtype)

        image_features = vision_tower(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            target_dtype = self.get_model().embed_tokens.weight.dtype
            cur_new_input_embeds = [
                x.to(device=self.device, dtype=target_dtype) for x in cur_new_input_embeds
            ]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        target_dtype = self.get_model().embed_tokens.weight.dtype

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            cur_new_embed = cur_new_embed.to(dtype=target_dtype)
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=target_dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=target_dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ============== LLaVA Qwen3 Model ==============

class LlavaQwen3Model(LlavaMetaModel, Qwen3Model):
    config_class = LlavaQwen3Config

    def __init__(self, config: LlavaQwen3Config):
        super(LlavaQwen3Model, self).__init__(config)


class LlavaQwen3ForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen3Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = LlavaQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


# Register model
AutoConfig.register("llava_qwen3", LlavaQwen3Config)
AutoModelForCausalLM.register(LlavaQwen3Config, LlavaQwen3ForCausalLM)
