#!/usr/bin/env python3
"""
Generate videos from images using HunyuanVideo-I2V model.
This script processes multiple input images and generates corresponding videos.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_models_available(model_base="./ckpts"):
    """
    Check if required models are available.
    
    Args:
        model_base: Path to model checkpoints
        
    Returns:
        Dictionary with model availability status
    """
    
    models_to_check = {
        "HunyuanVideo-I2V": os.path.join(model_base, "hunyuan-video-i2v-720p"),
        "Text Encoder I2V": os.path.join(model_base, "text_encoder_i2v"),
        "Text Encoder 2 (CLIP)": os.path.join(model_base, "text_encoder_2"),
        "VAE": os.path.join(model_base, "vae"),
    }
    
    status = {}
    for name, path in models_to_check.items():
        status[name] = os.path.exists(path)
    
    return status

def print_model_download_instructions():
    """Print instructions for downloading required models."""
    
    print("\n" + "="*70)
    print("📥 MODEL DOWNLOAD REQUIRED")
    print("="*70)
    print("\nThe following models need to be downloaded from HuggingFace:")
    print("\n1. First, install HuggingFace CLI:")
    print("   pip install huggingface-hub")
    print("\n2. Download HunyuanVideo-I2V model (this will download all necessary files):")
    print("   huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts")
    print("\n   Note: This downloads ~80GB of models. For faster downloads in China,")
    print("   you can use the mirror:")
    print("   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts")
    print("\n3. Download Text Encoder (LLaVA):")
    print("   cd ckpts && huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./text_encoder_i2v")
    print("\n4. Download CLIP model:")
    print("   huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2")
    print("\nAfter downloading all models, run this script again.")
    print("="*70 + "\n")

def generate_video_from_image(image_path, image_name, image_desc, output_dir, model_base="./ckpts"):
    """
    Generate a video from a single image using HunyuanVideo-I2V.
    
    Args:
        image_path: Path to the input image
        image_name: Name of the image file
        image_desc: Description of the image
        output_dir: Directory to save the output video
        model_base: Path to model checkpoints
    """
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"🎬 Generating video from: {image_desc}")
    print(f"{'='*60}")
    print(f"Image: {image_name}")
    print(f"Path: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command based on image type
    prompts = {
        "cat_headphones.png": "A cat wearing headphones in a digital cyber world with glowing effects and binary code",
        "midnight_meow_cafe.png": "A cozy cat sleeping peacefully in a warm room overlooking a rainy midnight city",
        "cozy_night_scene.png": "A tabby cat sleeping peacefully with toys and books on a windowsill at night",
    }
    
    prompt = prompts.get(image_name, f"A beautiful video generated from {image_desc}")
    
    cmd = [
        sys.executable,
        "sample_image2video.py",
        "--model", "HYVideo-T/2",
        "--model-base", model_base,
        "--save-path", output_dir,
        "--i2v-mode",
        "--i2v-image-path", image_path,
        "--i2v-resolution", "720p",
        "--i2v-stability",
        "--prompt", prompt,
        "--video-length", "129",
        "--infer-steps", "50",
        "--flow-reverse",
        "--flow-shift", "7.0",
        "--seed", "42",
        "--embedded-cfg-scale", "6.0",
        "--use-cpu-offload",
    ]
    
    print(f"\n⚙️  Command:")
    print(" ".join(cmd[:15]) + " \\\n    " + " \\\n    ".join(cmd[15:]))
    print()
    
    try:
        # Run the generation
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"\n✅ Successfully generated video for {image_desc}")
            return True
        else:
            print(f"\n❌ Failed to generate video for {image_desc}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating video: {str(e)}")
        return False


def main():
    """Main function to orchestrate video generation from multiple images."""
    
    # Define input images
    input_dir = Path("./input_images")
    output_base_dir = Path("./output_videos")
    model_base = "./ckpts"
    
    images_to_process = [
        ("cat_headphones.png", "Cat with Headphones"),
        ("midnight_meow_cafe.png", "Midnight Meow Café"),
        ("cozy_night_scene.png", "Cozy Night Scene"),
    ]
    
    print("\n" + "="*60)
    print("🚀 HunyuanVideo-I2V Video Generation Pipeline")
    print("="*60)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir.absolute()}")
        return False
    
    # Check if models are available
    print(f"\n🔍 Checking for required models...")
    model_status = check_models_available(model_base)
    
    all_available = all(model_status.values())
    print()
    for model_name, available in model_status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {model_name}")
    
    if not all_available:
        print_model_download_instructions()
        return False
    
    print(f"\n📁 Configuration:")
    print(f"   Input directory:  {input_dir.absolute()}")
    print(f"   Output directory: {output_base_dir.absolute()}")
    print(f"   Model directory:  {Path(model_base).absolute()}")
    print(f"   Images to process: {len(images_to_process)}")
    
    # Create main output directory
    output_base_dir.mkdir(exist_ok=True)
    
    # Process each image
    success_count = 0
    failed_count = 0
    results = []
    
    for image_file, image_desc in images_to_process:
        image_path = input_dir / image_file
        
        # Create specific output directory for this image
        image_output_dir = output_base_dir / image_file.replace(".png", "")
        
        # Generate video
        success = generate_video_from_image(
            str(image_path),
            image_file,
            image_desc,
            str(image_output_dir),
            model_base=model_base
        )
        
        if success:
            success_count += 1
            results.append((image_desc, "✅ Success"))
        else:
            failed_count += 1
            results.append((image_desc, "❌ Failed"))
        
        # Small delay between generations
        if image_file != images_to_process[-1][0]:
            print(f"\n⏳ Waiting 5 seconds before next generation...")
            time.sleep(5)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Generation Summary")
    print("="*60)
    for desc, status in results:
        print(f"{status} - {desc}")
    
    print(f"\n✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Output videos: {output_base_dir.absolute()}")
    print("="*60 + "\n")
    
    return failed_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
