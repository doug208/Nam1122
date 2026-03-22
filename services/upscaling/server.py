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
# Configuration for content-aware pre-processing
NOISE_THRESHOLD = 100  # Laplacian variance below this indicates noisy input
SHARPNESS_THRESHOLD = 500  # Laplacian variance above this indicates sharp input
MAX_SAMPLE_FRAMES = 10  # Maximum frames to sample for analysis
DENOISE_LUMA_SPATIAL = 2  # hqdn3d luma spatial strength (light denoising)
DENOISE_CHROMA_SPATIAL = 1  # hqdn3d chroma spatial strength
DENOISE_LUMA_TEMPORAL = 2  # hqdn3d luma temporal strength
DENOISE_CHROMA_TEMPORAL = 1  # hqdn3d chroma temporal strength

# Configuration for sharpness reduction
SHARPNESS_REDUCTION_KERNEL_SIZE = 3  # Gaussian blur kernel size for sharpness reduction
SHARPNESS_REDUCTION_SIGMA = 0.5  # Gaussian blur sigma for sharpness reduction

app = FastAPI()

# Module-level cache for loaded HAT-L models
_model_cache = {}

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    # output_file_upscaled: Optional[str] = None


# HAT-L Model URLs from Hugging Face (anchuang model repository)
HAT_L_MODEL_URLS = {
    "2": "https://huggingface.co/anchuang/HAT-L_SRx2_ImageNet-pretrain/resolve/main/HAT-L_SRx2_ImageNet-pretrain.pth",  # HAT-L 2x upscaling model
    "4": "https://huggingface.co/anchuang/HAT-L_SRx4_ImageNet-pretrain/resolve/main/HAT-L_SRx4_ImageNet-pretrain.pth"  # HAT-L 4x upscaling model
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


def get_or_load_model(scale_factor: str, device: torch.device):
    """
    Gets a cached HAT-L model or loads it from disk if not cached.
    
    Args:
        scale_factor: The upscaling factor ("2" or "4")
        device: The torch device to load the model on
        
    Returns:
        The loaded HAT-L model ready for inference.
    """
    cache_key = f"hat_l_{scale_factor}_{device}"
    if cache_key not in _model_cache:
        model_weights_path = get_hat_model_path(scale_factor)
        loader = ModelLoader()
        model_descriptor = loader.load_from_file(str(model_weights_path))
        model = model_descriptor.model.to(device)
        model.eval()
        _model_cache[cache_key] = model
    return _model_cache[cache_key]


def detect_noise_and_sharpness(video_path: Path, sample_count: int = MAX_SAMPLE_FRAMES) -> tuple[float, float]:
    """
    Detect noise and sharpness levels in a video using variance-of-Laplacian.
    
    Args:
        video_path: Path to the video file
        sample_count: Number of frames to sample for analysis
        
    Returns:
        tuple: (noise_score, sharpness_score) where:
            - noise_score: Average Laplacian variance (lower = more noise)
            - sharpness_score: Average Laplacian variance (higher = sharper)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0.0, 0.0
    
    # Sample frames evenly distributed throughout the video
    sample_indices = []
    if total_frames <= sample_count:
        sample_indices = list(range(total_frames))
    else:
        step = total_frames / sample_count
        sample_indices = [int(i * step) for i in range(sample_count)]
    
    laplacian_variances = []
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to grayscale for Laplacian analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (measure of sharpness/noise)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        laplacian_variances.append(variance)
    
    cap.release()
    
    if not laplacian_variances:
        return 0.0, 0.0
    
    avg_variance = sum(laplacian_variances) / len(laplacian_variances)
    
    # Noise score: lower variance indicates more noise
    # Sharpness score: higher variance indicates sharper image
    return avg_variance, avg_variance


def apply_denoising(input_path: Path, output_path: Path) -> bool:
    """
    Apply light denoising using FFmpeg hqdn3d filter.
    
    Args:
        input_path: Path to input video
        output_path: Path to output denoised video
        
    Returns:
        bool: True if denoising succeeded, False otherwise
    """
    denoise_cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", f"hqdn3d={DENOISE_LUMA_SPATIAL}:{DENOISE_CHROMA_SPATIAL}:{DENOISE_LUMA_TEMPORAL}:{DENOISE_CHROMA_TEMPORAL}",
        "-c:v", "libx264",
        "-crf", "23",  # Slightly higher quality for denoised output
        "-preset", "fast",
        "-c:a", "copy",  # Copy audio without re-encoding
        "-y",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            denoise_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300
        )
        return result.returncode == 0 and output_path.exists()
    except subprocess.TimeoutExpired:
        logger.warning("Denoising timed out")
        return False
    except Exception as e:
        logger.warning(f"Denoising failed: {e}")
        return False


def apply_sharpening_reduction(input_path: Path, output_path: Path) -> bool:
    """
    Apply sharpening reduction using FFmpeg unsharp filter with negative values.
    This reduces over-sharpening artifacts from HAT-L when the source is already sharp.
    
    Args:
        input_path: Path to input video (upscaled)
        output_path: Path to output video with reduced sharpening
        
    Returns:
        bool: True if sharpening reduction succeeded, False otherwise
    """
    # Use unsharp filter with negative values to reduce sharpening
    # luma_msize_x:luma_msize_y:luma_amount:chroma_msize_x:chroma_msize_y:chroma_amount
    # Negative amount values reduce sharpening
    sharpen_reduction_cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", "unsharp=3:3:-0.5:3:3:-0.3",  # Reduce luma and chroma sharpening
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
        "-c:a", "copy",  # Copy audio without re-encoding
        "-y",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            sharpen_reduction_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300
        )
        return result.returncode == 0 and output_path.exists()
    except subprocess.TimeoutExpired:
        logger.warning("Sharpening reduction timed out")
        return False
    except Exception as e:
        logger.warning(f"Sharpening reduction failed: {e}")
        return False


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

        # Step 1.5: Content-aware pre-processing (noise detection and optional denoising)
        print("Step 1.5: Analyzing video quality for content-aware pre-processing...")
        preprocessing_start = time.time()
        
        # Detect noise and sharpness levels
        noise_score, sharpness_score = detect_noise_and_sharpness(output_file_with_extra_frames)
        print(f"  Noise/Sharpness score (Laplacian variance): {noise_score:.2f}")
        print(f"  Noise threshold: {NOISE_THRESHOLD}, Sharpness threshold: {SHARPNESS_THRESHOLD}")
        
        # Determine if source is noisy (low Laplacian variance indicates noise)
        is_noisy = noise_score < NOISE_THRESHOLD
        # Determine if source is already sharp (high Laplacian variance indicates sharpness)
        is_sharp = sharpness_score > SHARPNESS_THRESHOLD
        
        print(f"  Is noisy: {is_noisy}, Is already sharp: {is_sharp}")
        
        # Apply denoising if video is noisy
        processed_file = output_file_with_extra_frames
        if is_noisy:
            print("  Noisy input detected. Applying light denoising with hqdn3d...")
            denoised_file = output_file_with_extra_frames.with_name(f"{input_file.stem}_denoised.mp4")
            
            if apply_denoising(output_file_with_extra_frames, denoised_file):
                print(f"  Denoising completed. Using denoised file: {denoised_file}")
                # Clean up the non-denoised file to save space
                if output_file_with_extra_frames.exists():
                    output_file_with_extra_frames.unlink()
                processed_file = denoised_file
            else:
                print("  Denoising failed, continuing with original file")
                if denoised_file.exists():
                    denoised_file.unlink()
        
        preprocessing_elapsed = time.time() - preprocessing_start
        print(f"Step 1.5 completed in {preprocessing_elapsed:.2f} seconds")
        
        # Store sharpness info for HAT-L post-processing adjustment
        # When source is already sharp, we apply sharpening reduction to avoid artifacts
        source_sharpness_level = "high" if is_sharp else "normal"
        print(f"  Source sharpness level: {source_sharpness_level}")

        # Step 2: Upscale video using HAT-L model
        print("Step 2: Upscaling video using HAT-L...")
        start_time = time.time()
        
        # Load HAT-L model using spandrel with pretrained weights (cached)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Get cached model or load from disk
        print(f"Loading HAT-L {scale_factor}x model with pretrained weights...")
        model = get_or_load_model(scale_factor, device)
        print(f"HAT-L {scale_factor}x model loaded successfully")
        
        # Process video frames
        temp_dir = output_file_upscaled.parent / f"{output_file_upscaled.stem}_frames"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract frames from video
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        extract_cmd = [
            "ffmpeg",
            "-i", str(processed_file),
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
        
        # Batch process frames with HAT-L for better GPU utilization
        # Process in chunks of 16 frames to balance memory usage and performance
        BATCH_SIZE = 16
        
        with torch.no_grad():
            num_batches = (len(frame_files) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, len(frame_files))
                batch_frame_paths = frame_files[start_idx:end_idx]
                
                # Load all frames in the batch
                batch_frames = []
                for frame_path in batch_frame_paths:
                    frame = cv2.imread(str(frame_path))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(frame)
                
                # Stack frames into a batch tensor (B, H, W, C) -> (B, C, H, W)
                batch_tensor = torch.from_numpy(np.stack(batch_frames)).float() / 255.0
                batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(device)
                
                # Upscale batch with HAT-L in a single forward pass
                upscaled_batch = model(batch_tensor)
                
                # Convert back to numpy and save each frame
                upscaled_batch = upscaled_batch.permute(0, 2, 3, 1).cpu().numpy()
                upscaled_batch = (upscaled_batch * 255.0).clip(0, 255).astype(np.uint8)
                
                for i, upscaled in enumerate(upscaled_batch):
                    frame_idx = start_idx + i
                    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                    output_frame_path = upscaled_frames_dir / f"frame_{frame_idx+1:08d}.png"
                    cv2.imwrite(str(output_frame_path), upscaled_bgr)
                
                print(f"Processed batch {batch_idx + 1}/{num_batches} ({end_idx}/{len(frame_files)} frames)...")
        
        # Encode upscaled frames back to video
        # If source was already sharp, apply sharpening reduction to avoid artifacts
        if is_sharp:
            print("  Source was already sharp. Applying sharpening reduction to avoid over-sharpening artifacts...")
            temp_upscaled_file = temp_dir / "upscaled_temp.mp4"
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
                str(temp_upscaled_file)
            ]
            
            encode_process = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if encode_process.returncode == 0 and temp_upscaled_file.exists():
                # Apply sharpening reduction
                if apply_sharpening_reduction(temp_upscaled_file, output_file_upscaled):
                    print(f"  Sharpening reduction applied. Final output: {output_file_upscaled}")
                    # Clean up temp file
                    if temp_upscaled_file.exists():
                        temp_upscaled_file.unlink()
                else:
                    print("  Sharpening reduction failed, using upscaled video without reduction")
                    # Rename temp file to output
                    temp_upscaled_file.rename(output_file_upscaled)
            else:
                print(f"  Video encoding failed: {encode_process.stderr.strip()}")
        else:
            # Normal encoding without sharpening reduction
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
        # Note: processed_file is either output_file_with_extra_frames or the denoised file
        # If denoising was applied, output_file_with_extra_frames was already deleted during denoising
        if processed_file.exists():
            processed_file.unlink()
            print(f"Intermediate file {processed_file} deleted.")
            
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


@app.on_event("startup")
async def startup_event():
    """Pre-load HAT-L models at startup to eliminate first-request overhead."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Pre-loading HAT-L models on device: {device}")
    get_or_load_model("2", device)
    get_or_load_model("4", device)
    logger.info("HAT-L models pre-loaded successfully")


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)