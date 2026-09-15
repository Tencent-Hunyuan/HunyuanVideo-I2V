# 🎬 HunyuanVideo-I2V Video Generation Guide

This guide explains how to generate 3 videos from the cat images using the HunyuanVideo-I2V model.

## 📋 Overview

The following 3 images have been prepared for video generation:
1. **Cat with Headphones** - A fluffy cat wearing headphones in a digital cyber environment
2. **Midnight Meow Café** - A cozy tabby cat sleeping in a warm room overlooking a rainy midnight city
3. **Cozy Night Scene** - A tabby cat sleeping peacefully with toys and books on a windowsill at night

## ⚙️ System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 24GB VRAM (RTX 3090, RTX 4090, A100, or similar)
  - For 720p resolution: ~24GB VRAM minimum
  - The script uses `--use-cpu-offload` to reduce memory usage
- **CPU**: Modern multi-core processor (8+ cores recommended)
- **RAM**: At least 32GB system RAM
- **Storage**: ~150GB for models + output space

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- cudnn library

## 📥 Step 1: Download Required Models

The models are hosted on HuggingFace and need to be downloaded before running the inference.

### 1.1 Install HuggingFace CLI

```bash
pip install huggingface-hub
```

### 1.2 Download Main Model (~80GB)

Run this command in the HunyuanVideo-I2V directory:

```bash
# Standard download
huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts

# If you're in China or experiencing slow downloads, use the mirror:
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts
```

**Note**: This downloads approximately 80GB of model files. Depending on your internet connection, this may take 30 minutes to several hours.

### 1.3 Download Text Encoder (LLaVA) (~18GB)

```bash
cd ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./text_encoder_i2v
```

### 1.4 Download CLIP Model (~340MB)

```bash
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
cd ..
```

### Expected Model Structure

After downloading, your `ckpts` directory should look like:
```
ckpts/
├── hunyuan-video-i2v-720p/
│   ├── transformers/
│   │   └── mp_rank_00_model_states.pt
│   ├── vae/
│   └── ...
├── text_encoder_i2v/
│   └── (LLaVA model files)
├── text_encoder_2/
│   └── (CLIP model files)
└── ...
```

## 🚀 Step 2: Run Video Generation

Once all models are downloaded, run the generation script:

```bash
python generate_videos_i2v.py
```

### What the Script Does

1. **Verifies Models**: Checks if all required models are available
2. **Processes Each Image**:
   - **Cat with Headphones**: Generates a video with digital effects
   - **Midnight Meow Café**: Generates a cozy, stable video
   - **Cozy Night Scene**: Generates a peaceful sleeping scene
3. **Saves Videos**: Outputs MP4 files to `./output_videos/` directory

### Generation Parameters Used

- **Resolution**: 720p
- **Video Length**: 129 frames (~5 seconds at 24fps)
- **Inference Steps**: 50
- **Stability Mode**: Enabled (generates stable, less dynamic videos)
- **Flow Shift**: 7.0 (optimized for stability)
- **Seed**: 42 (reproducible results)

## 📊 Video Output

Generated videos will be saved in:
```
output_videos/
├── cat_headphones/
│   └── [timestamp]_seed42_*.mp4
├── midnight_meow_cafe/
│   └── [timestamp]_seed42_*.mp4
└── cozy_night_scene/
    └── [timestamp]_seed42_*.mp4
```

Each video:
- Format: MP4 (H.264 codec)
- Resolution: 1280×720 (720p)
- Frame rate: 24 FPS
- Duration: ~5 seconds
- File size: ~100-150 MB per video

## ⏱️ Expected Runtime

- **Per Video Generation**: 30-60 minutes
- **Total for 3 Videos**: 2-3 hours (with waiting time between generations)
- This depends on:
  - GPU speed (A100 is faster than RTX 3090)
  - System load
  - Inference steps (50 steps is the default)

## 🎯 Advanced Options

### For More Dynamic Videos

If you want more dynamic, less stable videos, modify the script:

```python
"--flow-shift", "17.0",  # Change from 7.0 to 17.0
# And remove or comment out:
# "--i2v-stability",
```

### For Faster Generation (Lower Quality)

Reduce inference steps:
```python
"--infer-steps", "25",  # Change from 50 to 25
```

### For Different Seeds

Change the seed value to get different variations:
```python
"--seed", "12345",  # Change from 42 to any number
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

If you get CUDA out of memory errors:
1. The script already uses `--use-cpu-offload` to reduce memory
2. Close other applications
3. Restart Python/clear GPU cache
4. Consider using a GPU with more VRAM

### Models Not Found

If the script says models are missing:
1. Verify download completed without errors
2. Check directory structure matches expected format
3. Re-run the download commands

### Generation Fails

1. Check GPU is available: `nvidia-smi`
2. Verify CUDA drivers are installed
3. Check available disk space for output
4. Review error messages for specific issues

### Slow Generation

- This is normal! Generating high-quality videos takes time
- Each step of the diffusion process requires GPU computation
- With 50 inference steps, expect 30-60 minutes per video
- Faster GPUs (A100, H100) will be significantly faster than consumer GPUs

## 📝 Files Reference

- `generate_videos_i2v.py` - Main script to generate all videos
- `sample_image2video.py` - Individual image-to-video inference script
- `input_images/` - Directory containing the 3 input images
- `output_videos/` - Directory where generated videos are saved
- `ckpts/` - Directory for model checkpoints

## 🔗 Useful Links

- **HunyuanVideo-I2V GitHub**: https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V
- **HuggingFace Model Page**: https://huggingface.co/tencent/HunyuanVideo-I2V
- **Technical Report**: https://arxiv.org/abs/2412.03603
- **Playground**: https://video.hunyuan.tencent.com/

## ✅ Quick Checklist

- [ ] GPU with 24GB+ VRAM available
- [ ] CUDA and cuDNN properly installed
- [ ] Python 3.8+ installed
- [ ] HuggingFace CLI installed (`pip install huggingface-hub`)
- [ ] Models downloaded (~100GB total)
- [ ] 3 input images in `input_images/` directory
- [ ] Run `python generate_videos_i2v.py`
- [ ] Wait for videos to be generated
- [ ] Check `output_videos/` for results

---

**Happy video generation! 🎬✨**
