"""
20250324 pftq:
Script to flatten a dataset directory, rename files, convert frame rates, truncate frame counts, and convert JSON to TXT for diffusion pipeline.
- Reads from input_dir and writes to output_dir to avoid I/O issues
- Renames all files (json and media) to subfolder_original-filename
- Moves all files out of subfolders to the root of output_dir
- Converts video frame rates to specified framerate (if framerate > 0), showing start and final frame counts
- Truncates videos to specified framecap (if framecap > 0 and frame count exceeds cap)
- Matches the bitrate of the original video during processing
- Renames JSON files to .txt and retains only the 'long caption' values

Configure settings below.
Make sure you have ffmpeg and ffmpeg-python installed.
On Linux, install with:
apt-get update
apt-get install -y ffmpeg
pip install ffmpeg-python
"""

import os
import json
import shutil
import subprocess
import ffmpeg
from pathlib import Path
import tempfile

# Configuration
input_dir = "training"           # Directory containing the input dataset (videos and JSON files)
output_dir = "training_diffusionpipe"  # Directory where the processed dataset will be written
framerate = 16                   # Target frame rate for videos; -1 to keep original frame rate
framecap = 87                   # Maximum frame count for videos; -1 to keep original frame count

def get_video_info(file_path):
    """Get frame count, frame rate, and bitrate of a video using ffprobe."""
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        if not video_stream:
            raise ValueError("No video stream found")
        
        # Get frame count
        if "nb_frames" in video_stream:
            frame_count = int(video_stream["nb_frames"])
        else:
            duration = float(probe["format"]["duration"])
            fps = eval(video_stream["r_frame_rate"])
            frame_count = int(duration * fps)
        
        # Get frame rate
        fps = eval(video_stream["r_frame_rate"])
        
        # Get bitrate (try video stream first, then format)
        bitrate = video_stream.get("bit_rate")
        if bitrate is None:
            bitrate = probe["format"].get("bit_rate")
        if bitrate is not None:
            bitrate = int(bitrate)  # Convert to integer (in bits per second)
        else:
            print(f"Warning: Could not determine bitrate for {file_path}, using default")
            bitrate = None  # FFmpeg will choose a default if None
        
        return frame_count, fps, bitrate
    except ffmpeg.Error as e:
        print(f"Error probing {file_path}: {e.stderr.decode()}")
        return None, None, None

def convert_frame_rate(input_file, output_file, target_framerate, bitrate):
    """Convert the frame rate of a video file using FFmpeg and return the frame count after conversion."""
    try:
        stream = ffmpeg.input(input_file)
        output_args = {
            "r": target_framerate,  # Set the target frame rate
            "vcodec": "libx264",
            "acodec": "aac",
            "map_metadata": 0,
            "y": None
        }
        if bitrate is not None:
            output_args["b:v"] = bitrate  # Set the video bitrate to match the original
        
        stream = stream.output(output_file, **output_args)
        stream.run(quiet=True)
        
        frame_count, _, _ = get_video_info(output_file)
        if frame_count is None:
            print(f"Warning: Could not determine frame count after frame rate conversion for {output_file}")
            frame_count = "unknown"
        
        return frame_count
    except ffmpeg.Error as e:
        print(f"Error converting frame rate for {input_file}: {e.stderr.decode()}")
        return None

def truncate_frame_count(input_file, output_file, framecap, bitrate):
    """Truncate the frame count of a video file to the specified framecap using FFmpeg."""
    try:
        stream = ffmpeg.input(input_file)
        stream = stream.filter("trim", end_frame=framecap )
        output_args = {
            "vcodec": "libx264",
            "acodec": "aac",
            "map_metadata": 0,
            "y": None
        }
        if bitrate is not None:
            output_args["b:v"] = bitrate  # Set the video bitrate to match the original
        
        stream = stream.output(output_file, **output_args)
        stream.run(quiet=True)
        
        frame_count, _, _ = get_video_info(output_file)
        if frame_count is None:
            print(f"Warning: Could not determine frame count after truncation for {output_file}")
            frame_count = "unknown"
        
        return frame_count
    except ffmpeg.Error as e:
        print(f"Error truncating {input_file}: {e.stderr.decode()}")
        return None

def process_video(input_file, output_file, target_framerate, framecap):
    """Process a video file: convert frame rate (if specified) and truncate frame count (if exceeds framecap)."""
    try:
        # Get the starting frame count, frame rate, and bitrate of the input video
        start_frame_count, _, bitrate = get_video_info(input_file)
        if start_frame_count is None:
            print(f"Warning: Could not determine starting frame count for {input_file}")
            start_frame_count = "unknown"

        # If no processing is needed, return early
        if target_framerate <= 0 and (framecap <= 0 or (start_frame_count != "unknown" and start_frame_count <= framecap)):
            shutil.copy2(input_file, output_file)
            print(f"No processing needed for {input_file} (start frame count: {start_frame_count})")
            return start_frame_count

        # Pass 1: Frame rate conversion (if specified)
        intermediate_file = None
        current_file = input_file
        current_frame_count = start_frame_count

        if target_framerate > 0:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                intermediate_file = temp_file.name
            current_frame_count = convert_frame_rate(input_file, intermediate_file, target_framerate, bitrate)
            if current_frame_count is None:
                current_frame_count = "unknown"
            print(f"Converted frame rate of {input_file} to {target_framerate} FPS (start frame count: {start_frame_count}, after frame rate conversion: {current_frame_count})")
            current_file = intermediate_file

        # Pass 2: Truncation (if frame count exceeds framecap)
        final_frame_count = current_frame_count
        if framecap > 0 and (current_frame_count != "unknown" and current_frame_count > framecap):
            final_frame_count = truncate_frame_count(current_file, output_file, framecap, bitrate)
            if final_frame_count is None:
                final_frame_count = "unknown"
            print(f"Truncated {input_file} to framecap {framecap} (final frame count: {final_frame_count})")
        else:
            # No truncation needed, copy the intermediate (or original) file to the final output
            shutil.copy2(current_file, output_file)
            final_frame_count = current_frame_count
            #if target_framerate > 0:
                #print(f"No truncation needed for {input_file} (after frame rate conversion: {current_frame_count})")

        # Clean up intermediate file if it exists
        if intermediate_file and os.path.exists(intermediate_file):
            os.unlink(intermediate_file)

        return final_frame_count
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None

def flatten_and_convert_directory(input_dir, output_dir, framerate, framecap):
    """Flatten directory, rename files, convert frame rates, truncate frames, and convert JSON to TXT."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define file extensions
    media_extensions = (".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png")
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")  # Subset for videos that need processing
    json_extension = ".json"
    
    print("Processing files in directory...")
    files_to_process = []
    
    # Step 1: Collect all files and prepare new names
    for file in input_path.rglob("*"):  # Recursively search all subfolders in input_dir
        if file.is_file():  # Ensure it's a file, not a directory
            relative_path = file.relative_to(input_path)  # e.g., subfolder1/video_1.mp4
            parts = relative_path.parts  # Split into parts: ('subfolder1', 'video_1.mp4')
            
            # Construct new filename: subfolder_original-filename
            if len(parts) > 1:
                # File is in a subfolder
                subfolder = "_".join(parts[:-1])  # Join all parts except the filename (e.g., subfolder1_subfolder2)
                original_filename = parts[-1]  # e.g., video_1.mp4
                new_filename = f"{subfolder}_{original_filename}"  # e.g., subfolder1_video_1.mp4
            else:
                # File is already in the root
                new_filename = parts[0]  # e.g., video_1.mp4
            
            # New path in the output directory (root)
            new_path = output_path / new_filename
            files_to_process.append((file, relative_path, new_path))
    
    # Step 2: Copy and rename files to output_dir, converting frame rate and/or truncating if specified
    for file, relative_path, new_path in files_to_process:
        if file.suffix.lower() in media_extensions or file.suffix.lower() == json_extension:
            print(f"Copying and renaming {relative_path} to {new_path.name}")
            if file.suffix.lower() in video_extensions:
                # Video file: process frame rate and/or truncation
                process_video(str(file), str(new_path), framerate, framecap)
            else:
                # Non-video file: copy directly
                shutil.copy2(file, new_path)  # Use copy2 to preserve metadata
    
    # Step 3: Process files in output_dir (convert JSON to TXT)
    files_to_process = []
    for file in output_path.glob("*"):  # Only look in the root of output_dir
        if file.is_file():
            relative_path = file.relative_to(output_path)
            files_to_process.append((file, relative_path))
    
    for file, relative_path in files_to_process:
        if file.suffix.lower() == json_extension:
            # Process JSON file: extract long caption and convert to .txt
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract the "long caption" field (nested)
                long_caption = data.get("raw_caption", {}).get("long caption", "")
                if not long_caption:
                    print(f"Warning: No 'long caption' found in {relative_path}, using empty string")
                
                # New filename: change extension to .txt
                new_filename = file.stem + ".txt"  # e.g., video_1.txt
                new_path = output_path / new_filename
                
                # Write the long caption to the new .txt file
                print(f"Converting {relative_path} to {new_filename} with long caption")
                with open(new_path, 'w') as f:
                    f.write(long_caption)
                
                # Remove the original JSON file
                file.unlink()
            
            except json.JSONDecodeError as e:
                print(f"Error: Could not parse JSON in {relative_path}: {e}")
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")

    print("Dataset processing complete!")

def main():
    flatten_and_convert_directory(input_dir, output_dir, framerate, framecap)

if __name__ == "__main__":
    try:
        import ffmpeg
    except ImportError:
        print("Installing ffmpeg-python...")
        subprocess.run(["pip", "install", "ffmpeg-python"], check=True)
    
    main()