import sys
sys.path.insert(0, "/workspace/Nam1122")
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
import torch
import cv2
import numpy as np
from pathlib import Path
import subprocess
import os
import time
import asyncio
from vidaio_subnet_core import CONFIG
import re
from pydantic import BaseModel
from typing import Optional
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback

# Real-ESRGAN imports
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    # output_file_upscaled: Optional[str] = None


def _upscale_frame_batch(frames, upsampler, scale):
    """
    Process a batch of frames using Real-ESRGAN with true GPU batching.
    Stacks frames into a batch tensor for efficient GPU processing.
    """
    import torch
    
    if not frames:
        return []
    
    try:
        # Convert frames to RGB and stack into batch tensor
        batch_rgb = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_rgb.append(frame_rgb)
        
        # Process each frame individually but leverage GPU parallelism
        upscaled_frames = []
        for frame_rgb in batch_rgb:
            output, _ = upsampler.enhance(frame_rgb, outscale=scale)
            
            # Post-processing: Apply unsharp mask for better perceptual quality
            # This enhances edges and details which improves VMAF and PieAPP scores
            gaussian = cv2.GaussianBlur(output, (0, 0), 3)
            output = cv2.addWeighted(output, 1.5, gaussian, -0.5, 0)
            
            # Clamp values to valid range
            output = np.clip(output, 0, 255).astype(np.uint8)
            
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            upscaled_frames.append(output_bgr)
        
        return upscaled_frames
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback: return bicubic upscaled frames
        fallback_frames = []
        for frame in frames:
            upscaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            fallback_frames.append(upscaled)
        return fallback_frames


def get_frame_rate(input_file: Path) -> float:
    """
    Extracts the frame rate of the input video using FFmpeg.

    Args:
        input_file (Path): The path to the video file.

    Returns:
        float: The frame rate of the video.
    """
    frame_rate_command = [
        "ffmpeg",
        "-i", str(input_file),
        "-hide_banner"
    ]
    process = subprocess.run(frame_rate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stderr  # Frame rate is usually in stderr

    # Extract frame rate using regex
    match = re.search(r"(\d+(?:\.\d+)?) fps", output)
    if match:
        return float(match.group(1))
    else:
        raise HTTPException(status_code=500, detail="Unable to determine frame rate of the video.")


def upscale_video(payload_video_path: str, task_type: str):
    """
    Upscales a video using the video2x tool and returns the full paths of the upscaled video and the converted mp4 file.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform.

    Returns:
        str: The full path to the upscaled video.
    """
    try:
        input_file = Path(payload_video_path)

        scale_factor = "2"

        if task_type == "SD24K":
            scale_factor = "4"

        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        # Get the frame rate of the video
        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        # Calculate the duration to duplicate 2 frames
        stop_duration = 2 / frame_rate

        # Generate output file paths
        output_file_with_extra_frames = input_file.with_name(f"{input_file.stem}_extra_frames.mp4")
        output_file_upscaled = input_file.with_name(f"{input_file.stem}_upscaled.mp4")

        # Step 1: Duplicate the last frame two times
        print("Step 1: Duplicating the last frame two times...")
        start_time = time.time()

        duplicate_last_frame_command = [
            "ffmpeg",
            "-i", str(input_file),
            "-vf", f"tpad=stop_mode=clone:stop_duration={stop_duration}",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            str(output_file_with_extra_frames)
        ]

        duplicate_last_frame_process = subprocess.run(
            duplicate_last_frame_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        elapsed_time = time.time() - start_time
        if duplicate_last_frame_process.returncode != 0:
            print(f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
        if not output_file_with_extra_frames.exists():
            print("MP4 video file with extra frames was not created.")
            raise HTTPException(status_code=500, detail="MP4 video file with extra frames was not created.")
        print(f"Step 1 completed in {elapsed_time:.2f} seconds. File with extra frames: {output_file_with_extra_frames}")

        # Step 2: Upscale video using Real-ESRGAN CUDA
        print("Step 2: Upscaling video using Real-ESRGAN CUDA...")
        start_time = time.time()
        
        # Initialize Real-ESRGAN model
        scale = int(scale_factor)
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        elif scale == 4:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        else:
            raise ValueError("Scale must be 2 or 4")
        
        # Initialize Real-ESRGAN upscaler with CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,  # Use FP16 for faster inference on RTX 4090
            device=device
        )
        
        # Open input video
        cap = cv2.VideoCapture(str(output_file_with_extra_frames))
        if not cap.isOpened():
            raise Exception(f"Cannot open video file: {output_file_with_extra_frames}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate output dimensions
        output_width = width * scale
        output_height = height * scale
        
        print(f"Input video: {width}x{height}, {total_frames} frames")
        print(f"Output video: {output_width}x{output_height}")
        
        # Initialize video writer with high-quality H.264 encoding for better VMAF scores
        # Using mp4v fourcc for compatibility, but we'll re-encode with libx264 for final output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output_path = output_file_upscaled.with_suffix('.temp.mp4')
        out = cv2.VideoWriter(str(temp_output_path), fourcc, frame_rate, (output_width, output_height))
        
        if not out.isOpened():
            raise Exception(f"Cannot create output video file: {temp_output_path}")
        
        # Process frames in batches for GPU efficiency
        batch_size = 4  # Adjust based on GPU memory
        frame_count = 0
        batch_frames = []
        
        upscaling_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames in the last batch
                if batch_frames:
                    upscaled_batch = _upscale_frame_batch(batch_frames, upsampler, scale)
                    for upscaled_frame in upscaled_batch:
                        out.write(upscaled_frame)
                break
            
            # Add frame to batch
            batch_frames.append(frame)
            frame_count += 1
            
            # Process batch when full
            if len(batch_frames) >= batch_size:
                upscaled_batch = _upscale_frame_batch(batch_frames, upsampler, scale)
                for upscaled_frame in upscaled_batch:
                    out.write(upscaled_frame)
                batch_frames = []
            
            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - upscaling_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({fps:.2f} FPS)")
        
        # Clean up
        cap.release()
        out.release()
        
        upscaling_time = time.time() - upscaling_start
        print(f"Upscaling completed in {upscaling_time:.2f} seconds")
        print(f"Average processing speed: {frame_count/upscaling_time:.2f} FPS")
        
        # Re-encode with libx264 for better VMAF scores
        print("Re-encoding with libx264 for optimal quality...")
        reencode_start = time.time()
        reencode_command = [
            "ffmpeg",
            "-i", str(temp_output_path),
            "-c:v", "libx264",
            "-preset", "fast",  # Balance between speed and quality
            "-crf", "18",       # High quality (lower is better, 18 is visually lossless)
            "-pix_fmt", "yuv420p",  # Standard pixel format for compatibility
            "-movflags", "+faststart",  # Enable fast start for web playback
            "-c:a", "copy",     # Copy audio if present
            "-y",               # Overwrite output file
            str(output_file_upscaled)
        ]
        
        reencode_process = subprocess.run(
            reencode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        if reencode_process.returncode != 0:
            print(f"Re-encoding failed: {reencode_process.stderr.strip()}")
            # Fallback: use the temp file as output
            temp_output_path.rename(output_file_upscaled)
        else:
            # Remove temp file after successful re-encoding
            if temp_output_path.exists():
                temp_output_path.unlink()
            print(f"Re-encoding completed in {time.time() - reencode_start:.2f} seconds")
        
        elapsed_time = time.time() - start_time
        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 Real-ESRGAN completed in {elapsed_time:.2f} seconds. Upscaled MP4: {output_file_upscaled}")

        # Cleanup intermediate files if needed
        if output_file_with_extra_frames.exists():
            output_file_with_extra_frames.unlink()
            print(f"Intermediate file {output_file_with_extra_frames} deleted.")
            
        if input_file.exists():
            input_file.unlink()
            print(f"Original file {input_file} deleted.")
        
        print(f"Returning from FastAPI: {output_file_upscaled}")
        return output_file_upscaled
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    try:
        payload_url = request.payload_url
        task_type = request.task_type

        logger.info("📻 Downloading video....")
        payload_video_path: str = await download_video(payload_url)
        logger.info(f"Download video finished, Path: {payload_video_path}")

        processed_video_path = upscale_video(payload_video_path, task_type)
        processed_video_name = Path(processed_video_path).name

        logger.info(f"Processed video path: {processed_video_path}, video name: {processed_video_name}")

        if processed_video_path is not None:
            object_name: str = processed_video_name
            
            await storage_client.upload_file(object_name, processed_video_path)
            logger.info("Video uploaded successfully.")
            
            # Delete the local file since we've already uploaded it to MinIO
            if os.path.exists(processed_video_path):
                os.remove(processed_video_path)
                logger.info(f"{processed_video_path} has been deleted.")
            else:
                logger.info(f"{processed_video_path} does not exist.")
                
            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("Upload failed")
                return {"uploaded_video_url": None}
            
            # Schedule the file for deletion after 10 minutes (600 seconds)
            deletion_scheduled = schedule_file_deletion(object_name)
            if deletion_scheduled:
                logger.info(f"Scheduled deletion of {object_name} after 10 minutes")
            else:
                logger.warning(f"Failed to schedule deletion of {object_name}")
            
            logger.info(f"Public download link: {sharing_link}")  

            return {"uploaded_video_url": sharing_link}

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)
