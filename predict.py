#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse

import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def _is_qwen3_model(model_path_or_name):
    """Check if the model is a Qwen3 model based on path or name."""
    name_lower = model_path_or_name.lower() if model_path_or_name else ""
    return "qwen3" in name_lower or "qwen/qwen3" in name_lower


def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device="mps")

    # Check if using Qwen3 model for specific handling
    is_qwen3 = _is_qwen3_model(model_path) or _is_qwen3_model(model_name)

    # Construct prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("mps"))

    # Load and preprocess image
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Run inference
    with torch.inference_mode():
        # Qwen3-specific generation parameters
        gen_kwargs = dict(
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p if args.top_p else (0.8 if is_qwen3 else None),
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens if hasattr(args, 'max_new_tokens') else 512,
            use_cache=True,
        )

        # Add Qwen3-specific parameters
        if is_qwen3:
            gen_kwargs['top_k'] = 20

        output_ids = model.generate(input_ids, **gen_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # For Qwen3: remove thinking tags if present (when thinking mode was accidentally enabled)
        if is_qwen3 and "<think>" in outputs:
            import re
            outputs = re.sub(r'<think>.*?</think>', '', outputs, flags=re.DOTALL).strip()

        print(outputs)

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None, help="location of image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_3")
    parser.add_argument("--temperature", type=float, default=0.7)  # Qwen3 recommended
    parser.add_argument("--top_p", type=float, default=0.8)  # Qwen3 recommended
    parser.add_argument("--top_k", type=int, default=20)  # Qwen3 recommended
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    predict(args)
