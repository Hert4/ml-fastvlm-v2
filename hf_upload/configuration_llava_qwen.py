# Configuration for LLaVA Qwen3 model
# Upload this file to your HuggingFace repo for trust_remote_code=True

from transformers import Qwen2Config

try:
    from transformers import Qwen3Config
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False
    Qwen3Config = Qwen2Config  # Fallback


class LlavaQwen3Config(Qwen3Config if QWEN3_AVAILABLE else Qwen2Config):
    """Configuration class for LLaVA Qwen3 model."""

    model_type = "llava_qwen3"

    def __init__(
        self,
        mm_vision_tower=None,
        mm_hidden_size=None,
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_patch_merge_type="flat",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        image_aspect_ratio="pad",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.image_aspect_ratio = image_aspect_ratio
