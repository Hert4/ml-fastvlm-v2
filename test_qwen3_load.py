#!/usr/bin/env python
"""
Test script to verify Qwen3 model loading and configuration.
Run this to confirm Qwen3 support is properly set up.

Usage:
    python test_qwen3_load.py
"""

def test_qwen3_support():
    print("=" * 50)
    print("Testing Qwen3 Support for FastVLM")
    print("=" * 50)

    # Test 1: Check transformers version
    print("\n[1] Checking transformers version...")
    import transformers
    version = transformers.__version__
    print(f"    transformers version: {version}")

    major, minor = map(int, version.split(".")[:2])
    if major >= 4 and minor >= 51:
        print("    [OK] transformers >= 4.51.0")
    else:
        print("    [ERROR] transformers < 4.51.0 - Qwen3 not supported!")
        print("    Please run: pip install transformers>=4.51.0,<5.0.0")
        return False

    # Test 2: Check Qwen3 imports
    print("\n[2] Checking Qwen3 imports...")
    try:
        from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
        print("    [OK] Qwen3Config, Qwen3Model, Qwen3ForCausalLM imported")
    except ImportError as e:
        print(f"    [ERROR] Failed to import Qwen3 classes: {e}")
        return False

    # Test 3: Check LlavaQwen3 imports
    print("\n[3] Checking LlavaQwen3 imports...")
    try:
        from llava.model.language_model.llava_qwen import (
            LlavaQwen3Config,
            LlavaQwen3ForCausalLM,
            QWEN3_AVAILABLE
        )
        if QWEN3_AVAILABLE and LlavaQwen3ForCausalLM is not None:
            print("    [OK] LlavaQwen3Config, LlavaQwen3ForCausalLM imported")
        else:
            print("    [ERROR] Qwen3 classes are None - check transformers version")
            return False
    except ImportError as e:
        print(f"    [ERROR] Failed to import LlavaQwen3 classes: {e}")
        return False

    # Test 4: Check conversation templates
    print("\n[4] Checking conversation templates...")
    try:
        from llava.conversation import conv_templates, SeparatorStyle
        if "qwen_3" in conv_templates:
            print("    [OK] qwen_3 template found")
            conv = conv_templates["qwen_3"]
            print(f"    Version: {conv.version}")
            print(f"    Sep style: {conv.sep_style}")
        else:
            print("    [ERROR] qwen_3 template not found")
            return False

        if hasattr(SeparatorStyle, 'QWEN_3'):
            print("    [OK] QWEN_3 separator style exists")
        else:
            print("    [ERROR] QWEN_3 separator style not found")
            return False
    except ImportError as e:
        print(f"    [ERROR] Failed to import conversation: {e}")
        return False

    # Test 5: Load Qwen3 config from HuggingFace (optional - requires network)
    print("\n[5] Testing Qwen3 model config loading...")
    try:
        from transformers import AutoConfig
        model_name = "Qwen/Qwen3-0.6B"
        print(f"    Loading config from: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        print(f"    model_type: {config.model_type}")
        print(f"    hidden_size: {config.hidden_size}")
        print(f"    vocab_size: {config.vocab_size}")

        if config.model_type == "qwen3":
            print("    [OK] Config loaded successfully")
        else:
            print(f"    [WARNING] Unexpected model_type: {config.model_type}")
    except Exception as e:
        print(f"    [SKIP] Could not load config (network issue?): {e}")

    # Test 6: Test tokenizer with enable_thinking=False
    print("\n[6] Testing Qwen3 tokenizer...")
    try:
        from transformers import AutoTokenizer
        model_name = "Qwen/Qwen3-0.6B"
        print(f"    Loading tokenizer from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        messages = [{"role": "user", "content": "Hello"}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Important for Qwen3 non-reasoning mode
        )
        print(f"    Template output (first 100 chars): {ids[:100]}...")
        print("    [OK] Tokenizer works with enable_thinking=False")
    except Exception as e:
        print(f"    [SKIP] Could not test tokenizer: {e}")

    print("\n" + "=" * 50)
    print("All critical tests passed!")
    print("Qwen3 support is properly configured.")
    print("=" * 50)

    # Print model size mapping
    print("\nQwen3 Model Size Mapping:")
    print("  Qwen2-0.5B  -> Qwen3-0.6B  (hidden_size=1024)")
    print("  Qwen2-1.5B  -> Qwen3-1.7B  (hidden_size=2048)")
    print("  Qwen2-7B    -> Qwen3-8B    (hidden_size=4096)")

    return True


if __name__ == "__main__":
    import sys
    success = test_qwen3_support()
    sys.exit(0 if success else 1)
