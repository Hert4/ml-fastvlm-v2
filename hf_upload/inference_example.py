# ============================================
# Belle-VLM Inference Example
# After uploading custom code to HuggingFace
# ============================================

# Install: pip install transformers torch torchvision pillow accelerate

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============================================
# CONFIG
# ============================================
MODEL_ID = "beyoru/Belle-VLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# LOAD MODEL & TOKENIZER
# ============================================
print(f"Loading model from {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("Model loaded!")

# ============================================
# IMAGE PROCESSING
# ============================================
from torchvision import transforms

def process_image(image, size=384):
    """Process image for model input."""
    if isinstance(image, str):
        image = Image.open(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Expand to square
    w, h = image.size
    if w != h:
        new_size = max(w, h)
        new_img = Image.new('RGB', (new_size, new_size), (128, 128, 128))
        new_img.paste(image, ((new_size - w) // 2, (new_size - h) // 2))
        image = new_img

    # Transform
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    return transform(image)

# ============================================
# INFERENCE FUNCTION
# ============================================
IMAGE_TOKEN_INDEX = -200

def tokenize_with_image(prompt, tokenizer):
    """Tokenize prompt with image token."""
    chunks = prompt.split("<image>")
    tokens = []
    for i, chunk in enumerate(chunks):
        chunk_ids = tokenizer.encode(chunk, add_special_tokens=(i == 0))
        tokens.extend(chunk_ids)
        if i < len(chunks) - 1:
            tokens.append(IMAGE_TOKEN_INDEX)
    return torch.tensor(tokens, dtype=torch.long)


def ask_vlm(image, question, max_tokens=512, temperature=0.7):
    """
    Ask Belle-VLM about an image.

    Args:
        image: PIL Image or path to image
        question: Question in Vietnamese or English
        max_tokens: Max response length
        temperature: 0 = deterministic, higher = more creative

    Returns:
        Response string
    """
    # Process image
    if isinstance(image, str):
        pil_image = Image.open(image)
    else:
        pil_image = image

    image_tensor = process_image(pil_image)

    # Build prompt (Qwen3 format)
    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
{question}<|im_end|>
<|im_start|>assistant
"""

    # Tokenize
    input_ids = tokenize_with_image(prompt, tokenizer).unsqueeze(0).to(model.device)

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device=model.device),
            image_sizes=[pil_image.size],
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.8 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    # Test with a sample image
    # image = Image.open("test.jpg")
    # response = ask_vlm(image, "Mô tả hình ảnh này.")
    # print(response)

    print("\n" + "="*50)
    print("Belle-VLM Ready!")
    print("="*50)
    print("\nUsage:")
    print('  response = ask_vlm(image, "Mô tả hình ảnh này.")')
    print('  response = ask_vlm("path/to/image.jpg", "Trong hình có gì?")')
    print("\nParameters:")
    print("  - image: PIL Image or file path")
    print("  - question: Your question (Vietnamese/English)")
    print("  - max_tokens: Max response length (default: 512)")
    print("  - temperature: 0=deterministic, 0.7=creative (default: 0.7)")
