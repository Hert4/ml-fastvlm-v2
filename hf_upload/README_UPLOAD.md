# Files to Upload to HuggingFace

Upload these files to your HuggingFace model repo (`beyoru/Belle-VLM`) to enable `trust_remote_code=True`.

## Required Files

1. **configuration_llava_qwen.py** - Config class
2. **modeling_llava_qwen.py** - Model class

## How to Upload

### Option 1: Using huggingface_hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload files
api.upload_file(
    path_or_fileobj="configuration_llava_qwen.py",
    path_in_repo="configuration_llava_qwen.py",
    repo_id="beyoru/Belle-VLM",
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="modeling_llava_qwen.py",
    path_in_repo="modeling_llava_qwen.py",
    repo_id="beyoru/Belle-VLM",
    repo_type="model",
)
```

### Option 2: Using Git

```bash
# Clone your repo
git clone https://huggingface.co/beyoru/Belle-VLM
cd Belle-VLM

# Copy files
cp /path/to/configuration_llava_qwen.py .
cp /path/to/modeling_llava_qwen.py .

# Push
git add .
git commit -m "Add custom model code for trust_remote_code"
git push
```

### Option 3: Web Upload

1. Go to https://huggingface.co/beyoru/Belle-VLM
2. Click "Files and versions"
3. Click "Add file" -> "Upload files"
4. Upload both files

## Update config.json

After uploading, update your `config.json` to include:

```json
{
  "auto_map": {
    "AutoConfig": "configuration_llava_qwen.LlavaQwen3Config",
    "AutoModelForCausalLM": "modeling_llava_qwen.LlavaQwen3ForCausalLM"
  },
  ...
}
```

## After Upload

Then you can load the model with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "beyoru/Belle-VLM",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "beyoru/Belle-VLM",
    trust_remote_code=True
)
```

## Note

The vision tower in this simplified version is a stub. For full functionality with vision, you still need to either:
1. Use the full ml-fastvlm repo
2. Or ensure vision weights are embedded in the model checkpoint
