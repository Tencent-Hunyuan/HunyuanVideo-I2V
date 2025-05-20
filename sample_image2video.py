import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

# 20250329 pftq: For additional features ffmpeg video, saving prompt to comments metadata
import subprocess
import tempfile
import shutil
import sys
import numpy as np  # For frame conversion
import torch
import random
import imageio_ffmpeg

def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    ################# functions for better video files #######################

    # 20250328 pftq: Saving to mp4 comments metadata with ffmpeg if available
    def check_ffmpeg_installed():
        """Check if FFmpeg is installed and available in PATH."""
        try:
            # Run 'ffmpeg -version' to check if FFmpeg is accessible
            result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return True
            return False
        except FileNotFoundError:
            return False
    # FFmpeg-based video saving with bitrate control
    def save_video_with_ffmpeg(frames, output_path, fps, bitrate_mbps, metadata_comment=None):
        """
        Save a tensor or list of video frames to an MP4 file using FFmpeg.
        
        Args:
            frames: Tensor of shape (T, H, W, C) or list of (H, W, C) frames
            output_path: Path to save the MP4 file
            fps: Frames per second
            bitrate_mbps: Bitrate in Mbps
            metadata_comment: Optional metadata comment to embed in the video
        """
        # Convert tensor to NumPy if needed
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        # If frames is a single tensor (T, H, W, C), convert to list of (H, W, C)
        if len(frames.shape) == 4:
            T, H, W, C = frames.shape
            frames = [frames[i] for i in range(T)]
        elif isinstance(frames, (list, tuple)):
            H, W, C = frames[0].shape
        else:
            raise ValueError(f"Expected frames to be a tensor (T, H, W, C) or list of (H, W, C), got {frames.shape}")

        # Ensure frames are in RGB [0, 255] uint8 format
        processed_frames = []
        for frame in frames:
            frame = np.array(frame)
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)  # Scale from [0, 1] to [0, 255]
            elif frame.max() > 255:
                frame = np.clip(frame, 0, 255).astype(np.uint8)  # Clip to [0, 255]
            else:
                frame = frame.astype(np.uint8)  # Already in [0, 255]
            processed_frames.append(frame)
    
        height, width, _ = processed_frames[0].shape
        bitrate = f"{bitrate_mbps}M"
    
        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-loglevel", "verbose",  # Add verbose logging for debugging
        ]
        if metadata_comment:
            cmd.extend(["-metadata", f"comment={metadata_comment}"])
        cmd.append(output_path)
    
        # Pipe frames to FFmpeg in binary mode
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in processed_frames:
                process.stdin.write(frame.tobytes())  # Write raw bytes
            process.stdin.close()
            # Wait for FFmpeg to finish and capture stderr
            stderr_output = process.stderr.read().decode()
            return_code = process.wait()
            if return_code != 0:
                print(f"FFmpeg error: {stderr_output}")
                raise RuntimeError(f"FFmpeg failed with return code {return_code}: {stderr_output}")
            else:
                print(f"Video saved to {output_path} with FFmpeg")
        except Exception as e:
            print(f"Error in save_video_with_ffmpeg: {e}")
            # If process is still alive, try to capture stderr
            if 'process' in locals():
                stderr_output = process.stderr.read().decode() if process.stderr else "No stderr captured"
                print(f"FFmpeg stderr: {stderr_output}")
            raise

    def set_mp4_comments_imageio_ffmpeg(input_file, comments):
        try:
            # Get the path to the bundled FFmpeg binary from imageio-ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Check if input file exists
            if not os.path.exists(input_file):
                print(f"Error: Input file {input_file} does not exist")
                return False
                
            # Create a temporary file path
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            
            # FFmpeg command using the bundled binary
            command = [
                ffmpeg_path,                   # Use imageio-ffmpeg's FFmpeg
                '-i', input_file,              # input file
                '-metadata', f'comment={comments}',  # set comment metadata
                '-c:v', 'copy',                # copy video stream without re-encoding
                '-c:a', 'copy',                # copy audio stream without re-encoding
                '-y',                          # overwrite output file if it exists
                temp_file                      # temporary output file
            ]
            
            # Run the FFmpeg command
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                # Replace the original file with the modified one
                shutil.move(temp_file, input_file)
                print(f"Successfully added comments to {input_file}")
                return True
            else:
                # Clean up temp file if FFmpeg fails
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                print(f"Error: FFmpeg failed with message:\n{result.stderr}")
                return False
                
        except Exception as e:
            # Clean up temp file in case of other errors
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
            print(f"Error saving prompt to video metadata, ffmpeg may be required.")
            return False
            
    def reconstruct_command_line(args):
        cmd_parts = [sys.argv[0]]  # Start with script name
        args_dict = vars(args)  # Convert args to dict
        
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith("--"):
                key = arg[2:].replace('-', '_')  # Normalize hyphens to underscores
                if key in args_dict:
                    value = args_dict[key]
                    if isinstance(value, bool):
                        if value:
                            cmd_parts.append(arg)  # Boolean flag
                        i += 1
                    elif isinstance(value, list):  # Handle list values
                        # Assume the next few args in sys.argv are the list values
                        list_values = []
                        j = i + 1
                        while j < len(sys.argv) and not sys.argv[j].startswith("--"):
                            list_values.append(sys.argv[j])
                            j += 1
                        cmd_parts.append(f"{arg} {' '.join(str(v) for v in value)}")
                        i = j  # Skip past the list values
                    else:
                        # Check if there's a next value in sys.argv
                        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                            next_val = sys.argv[i + 1]
                            if isinstance(value, str):
                                cmd_parts.append(f'{arg} "{value}"')  # Quote strings
                            else:
                                cmd_parts.append(f"{arg} {value}")    # No quotes for numbers
                            i += 2
                        else:
                            # Use parsed value if no next arg in sys.argv
                            if isinstance(value, str):
                                cmd_parts.append(f'{arg} "{value}"')
                            else:
                                cmd_parts.append(f"{arg} {value}")
                            i += 1
                else:
                    # Handle unknown flags by adding them as-is
                    cmd_parts.append(arg)
                    i += 1
            else:
                # Handle positional arguments or unexpected input
                cmd_parts.append(arg)
                i += 1
    
        # Second pass: Add remaining args not in sys.argv
        seen_keys = {arg[2:].replace('-', '_') for arg in sys.argv if arg.startswith("--")}  # Normalize keys
        for key, value in vars(args).items():
            if key not in seen_keys:
                arg_name = f"--{key.replace('_', '-')}"  # Convert back to hyphenated form
                if isinstance(value, bool):
                    if value:
                        cmd_parts.append(arg_name)
                elif isinstance(value, list):
                    cmd_parts.append(f"{arg_name} {' '.join(str(v) for v in value)}")
                else:
                    if isinstance(value, str):
                        cmd_parts.append(f'{arg_name} "{value}"')
                    else:
                        cmd_parts.append(f"{arg_name} {value}")
        
        # Join with backslash and newline, treating each cmd_parts entry as a full argument
        if len(cmd_parts) > 1:
            return " \\\n".join(cmd_parts)
        return cmd_parts[0]  # Single arg case

    #####################################################################################

    for idx in range(args.batch_size): # 20250224 pftq: implemented --batch-size

        # 20250307 pftq: optional --variety-batch feature
        cfgdelta = 0
        stepsdelta = 0
        embeddeddelta=0
        variety_range=10
        if idx > 0: # reset the seed so the whole batch isn't the same video
            args.seed = None
        if args.variety_batch:
            if args.cfg_scale>1:
                variety_range = int(max(4, (20-args.cfg_scale)/2))
                cfgdelta = (idx % variety_range)*2
            else:
                variety_range = int(max(4, (20-args.embedded_cfg_scale)/2))
                embeddeddelta = (idx % variety_range)*2
            stepsdelta = int(idx // variety_range) * 10
            if stepsdelta>125:
                stepsdelta = 125
    
        # Start sampling
        # TODO: batch inference check
        
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps+stepsdelta,
            guidance_scale=args.cfg_scale+cfgdelta,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=1,
            embedded_guidance_scale=args.embedded_cfg_scale+embeddeddelta,
            i2v_mode=args.i2v_mode,
            i2v_resolution=args.i2v_resolution,
            i2v_image_path=args.i2v_image_path,
            i2v_condition_type=args.i2v_condition_type,
            i2v_stability=args.i2v_stability,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
        )
        samples = outputs['samples']
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    
                # 20250328 pftq: More useful filename and also updated cfg/steps info if using --variety-batch
                newcfg = args.cfg_scale+cfgdelta
                newsteps = args.infer_steps+stepsdelta
                newembedded = args.embedded_cfg_scale+embeddeddelta
                loraName=""
                if args.use_lora:
                    loraName = "_"+args.lora_path.replace("hunyuan_", "").replace("_", "-").replace("./", "lora-").replace(".safetensors", "")
    
                gpuInName = ""
                if args.ulysses_degree > 1 or args.ring_degree > 1:
                    gpuCount = args.ulysses_degree *  args.ring_degree
                    gpuInName = f'_{gpuCount}xGPU'
                
                cur_save_path = f"{save_path}/{time_flag}_hunyuani2v_{args.i2v_resolution}-{args.video_length}f_cfg{newcfg}_steps{newsteps}_embedded{newembedded}_flow{args.flow_shift}_seed{outputs['seeds'][i]}_stable-{args.i2v_mode}{loraName}{gpuInName}_{outputs['prompts'][i][:20].replace('/','')}_{idx}.mp4"

                # temporarily set the arg seed for saving to video file metadata
                inputSeed = args.seed
                args.seed = outputs['seeds'][i]
                
                if check_ffmpeg_installed(): # 20250329 pftq: higher quality bitrate if ffmpeg is installed
                    save_video_with_ffmpeg(samples[i].permute(1, 2, 3, 0), output_path=cur_save_path, fps=24, bitrate_mbps=15, metadata_comment=reconstruct_command_line(args))
                else:
                    logger.info("FFMPEG not installed, falling back to default video encoder... ")
                    sample = samples[i].unsqueeze(0)
                    save_videos_grid(sample, cur_save_path, fps=24)
                    # 20250328 pftq: Save prompt to video comments metadata
                    logger.info('Saving commandline prompt to video comments metadata...')
                    set_mp4_comments_imageio_ffmpeg(cur_save_path, reconstruct_command_line(args))

                args.seed = inputSeed
                    
                logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
