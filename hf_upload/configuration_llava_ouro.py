# Configuration for LLaVA Ouro model
# Upload this file to your HuggingFace repo for trust_remote_code=True

from transformers import PretrainedConfig


class LlavaOuroConfig(PretrainedConfig):
    """Configuration class for LLaVA Ouro model (LoopLM architecture)."""

    model_type = "llava_ouro"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Ouro base config
        vocab_size=49152,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=65536,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        # Ouro LoopLM specific
        total_ut_steps=4,
        early_exit_threshold=1.0,
        # LLaVA multimodal config
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
        # Ouro base config
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        # Ouro LoopLM specific
        self.total_ut_steps = total_ut_steps
        self.early_exit_threshold = early_exit_threshold

        # Layer types
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * num_hidden_layers

        # LLaVA multimodal config
        self.mm_vision_tower = mm_vision_tower
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.image_aspect_ratio = image_aspect_ratio

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
