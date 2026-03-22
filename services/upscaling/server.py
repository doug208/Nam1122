from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import os
from fastapi.responses import JSONResponse
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
import torch
import cv2
import numpy as np
from spandrel import ModelLoader, ImageModelDescriptor
import urllib.request
import shutil

app = FastAPI()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    # output_file_upscaled: Optional[str] = None


# HAT-L Model URLs from Hugging Face (anchuang model repository)
HAT_L_MODEL_URLS = {
    "2": "https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/main/HAT-L_SRx4_ImageNet-pretrain.pth",  # Using 4x model as fallback
    "4": "https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/main/HAT-L_SRx4_ImageNet-pretrain.pth"
}


def get_hat_model_path(scale_factor: str) -> Path:
    """
    Downloads HAT-L pretrained weights if not present and returns the local path.
    
    Args:
        scale_factor: The upscaling factor ("2" or "4")
        
    Returns:
        Path to the local HAT-L model weights file.
    """
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_filename = f"HAT-L_SRx{scale_factor}_ImageNet-pretrain.pth"
    model_path = models_dir / model_filename
    
    if not model_path.exists():
        if scale_factor not in HAT_L_MODEL_URLS:
            raise ValueError(f"Unsupported scale factor: {scale_factor}. Supported: {list(HAT_L_MODEL_URLS.keys())}")
        
        url = HAT_L_MODEL_URLS[scale_factor]
        print(f"Downloading HAT-L {scale_factor}x pretrained weights from {url}...")
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            if block_num % 100 == 0:  # Print every 100 blocks to avoid spam
                print(f"  Download progress: {percent:.1f}%")
        
        urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
        print(f"HAT-L {scale_factor}x weights downloaded to {model_path}")
    else:
        print(f"Using cached HAT-L {scale_factor}x weights from {model_path}")
    
    return model_path


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
    Upscales a video using the HAT-L (Hybrid Attention Transformer Large) model.
    
    This function processes video frames individually using HAT-L via the spandrel
    library, achieving superior perceptual quality compared to Real-ESRGAN.
    The implementation is optimized for NVIDIA RTX 4090 hardware.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform (e.g., "SD24K" for 4x).

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

        # Step 2: Upscale video using HAT-L model
        print("Step 2: Upscaling video using HAT-L...")
        start_time = time.time()
        
        # Load HAT-L model using spandrel with pretrained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Download and load pretrained HAT-L weights
        model_weights_path = get_hat_model_path(scale_factor)
        
        # Load model with pretrained weights using spandrel ModelLoader
        print(f"Loading HAT-L {scale_factor}x model with pretrained weights...")
        loader = ModelLoader()
        model_descriptor = loader.load_from_file(str(model_weights_path))
        model = model_descriptor.model.to(device)
        model.eval()
        print(f"HAT-L {scale_factor}x model loaded successfully")
        
        # Process video frames
        temp_dir = output_file_upscaled.parent / f"{output_file_upscaled.stem}_frames"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract frames from video
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        extract_cmd = [
            "ffmpeg",
            "-i", str(output_file_with_extra_frames),
            "-q:v", "2",  # High quality
            str(frames_dir / "frame_%08d.png")
        ]
        subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Get list of frames
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise HTTPException(status_code=500, detail="No frames extracted from video")
        
        print(f"Processing {len(frame_files)} frames with HAT-L...")
        
        # Process frames with HAT-L
        upscaled_frames_dir = temp_dir / "upscaled_frames"
        upscaled_frames_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for i, frame_path in enumerate(frame_files):
                # Read frame
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Upscale with HAT-L
                upscaled_tensor = model(frame_tensor)
                
                # Convert back to numpy
                upscaled = upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                upscaled = (upscaled * 255.0).clip(0, 255).astype(np.uint8)
                upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                
                # Save upscaled frame
                output_frame_path = upscaled_frames_dir / f"frame_{i+1:08d}.png"
                cv2.imwrite(str(output_frame_path), upscaled)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(frame_files)} frames...")
        
        # Encode upscaled frames back to video
        encode_cmd = [
            "ffmpeg",
            "-framerate", str(frame_rate),
            "-i", str(upscaled_frames_dir / "frame_%08d.png"),
            "-c:v", "libx265",
            "-preset", "slow",
            "-crf", "20",
            "-profile:v", "main",
            "-pix_fmt", "yuv420p",
            "-sar", "1:1",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",
            "-y",  # Overwrite output
            str(output_file_upscaled)
        ]
        
        encode_process = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Cleanup temporary directories
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        elapsed_time = time.time() - start_time
        
        if encode_process.returncode != 0:
            print(f"Video encoding failed: {encode_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Video encoding failed: {encode_process.stderr.strip()}")
        if not output_file_upscaled.exists():
            print("Upscaled MP4 video file was not created.")
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

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