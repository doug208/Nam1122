#!/usr/bin/env python3
"""
Fixed Real-ESRGAN Upscaling Service with HD24K Support

Key Fixes:
- Added support for HD24K task type
- Better error handling and logging
- Memory monitoring and self-recovery
"""

import os
# Set CUDA memory configuration before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
import gc
import psutil

# Real-ESRGAN imports
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI()

# Configuration
MEMORY_THRESHOLD_GB = 20  # If GPU memory exceeds this, use bicubic fallback
TILE_SIZE = 128  # Smaller tiles for better memory management
BATCH_SIZE = 1  # Process one frame at a time to control memory

# Model cache
_model_cache = {}

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    return 0

def cleanup_memory():
    """Aggressive memory cleanup."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")

def _get_upsampler(scale):
    """Get or create a cached Real-ESRGAN upsampler with memory monitoring."""
    global _model_cache
    
    # Check GPU memory before loading
    gpu_memory_mb = get_gpu_memory_usage()
    gpu_memory_gb = gpu_memory_mb / 1024
    
    if gpu_memory_gb > MEMORY_THRESHOLD_GB:
        logger.warning(f"GPU memory high ({gpu_memory_gb:.1f}GB), using bicubic fallback")
        return None  # Signal to use bicubic
    
    if scale not in _model_cache:
        logger.info(f"Loading RealESRGAN_x{scale}plus model...")
        
        # Use smaller model architecture
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        else:
            # Use the anime model which is more memory efficient
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.1/RealESRGAN_x4plus_anime_6B.pth'
        
        try:
            # Use smaller tile size and enable half precision
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                dni_weight=None,
                model=model,
                tile=TILE_SIZE,  # Smaller tiles
                tile_pad=10,
                pre_pad=0,
                half=True,  # Use FP16
                device=torch.device("cuda")
            )
            _model_cache[scale] = upsampler
            logger.info(f"RealESRGAN_x{scale}plus model loaded successfully")
            
            # Cleanup after loading
            cleanup_memory()
            
        except Exception as e:
            logger.error(f"Failed to load RealESRGAN model: {e}")
            return None
    
    return _model_cache.get(scale)

def _upscale_frame_batch(frames, upsampler, scale):
    """GPU batch upscaling with memory monitoring and fallback."""
    if not frames:
        return []
    
    upscaled_frames = []
    
    # Check GPU memory before processing
    gpu_memory_gb = get_gpu_memory_usage() / 1024
    use_gpu = gpu_memory_gb <= MEMORY_THRESHOLD_GB and upsampler is not None
    
    if not use_gpu:
        logger.warning(f"Using bicubic fallback (GPU memory: {gpu_memory_gb:.1f}GB)")
        # Fall back to bicubic interpolation
        for frame in frames:
            h, w = frame.shape[:2]
            upscaled_frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            upscaled_frames.append(upscaled_frame)
        return upscaled_frames
    
    # Process frames in batches
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i + BATCH_SIZE]
        
        try:
            # Convert to tensor
            batch_tensor = torch.from_numpy(np.stack(batch)).to(
                device=torch.device("cuda"),
                dtype=torch.float16,
                non_blocking=True
            )
            
            # Process batch
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = upsampler.model(batch_tensor)
            
            # Convert back
            output_np = output.detach().cpu().numpy()
            
            for frame_data in output_np:
                frame_hwc = np.transpose(frame_data, (1, 2, 0))
                frame_uint8 = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
                upscaled_frames.append(frame_uint8)
            
            # Cleanup immediately
            del batch_tensor, output, output_np
            cleanup_memory()
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM at batch {i}, falling back to bicubic")
                for frame in batch:
                    h, w = frame.shape[:2]
                    upscaled_frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                    upscaled_frames.append(upscaled_frame)
                # Force cleanup and switch to bicubic for remaining frames
                cleanup_memory()
                use_gpu = False
            else:
                raise
    
    return upscaled_frames

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str

@app.post("/upscale-video")
async def upscale_video(request: UpscaleRequest):
    """Upscale a video file with robust memory management."""
    try:
        # Parse task type - handle both upscale_Xx and HD24K formats
        scale_match = re.search(r'upscale_(\d+)x', request.task_type)
        if scale_match:
            scale = int(scale_match.group(1))
        elif request.task_type == 'HD24K':
            # HD24K is a specific task type, default to 4x upscaling
            scale = 4
        else:
            raise HTTPException(status_code=400, detail=f"Invalid task_type format: {request.task_type}. Expected 'upscale_2x', 'upscale_4x', or 'HD24K'")
        
        if scale not in [2, 4]:
            raise HTTPException(status_code=400, detail="Scale must be 2 or 4")
        
        # Get upsampler (may return None if memory is low)
        upsampler = _get_upsampler(scale)
        
        # Download video
        temp_input_path_str = await download_video(request.payload_url)
        temp_input_path = Path(temp_input_path_str)
        
        # Extract frames
        cap = cv2.VideoCapture(str(temp_input_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        temp_input_path.unlink()
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        # Upscale frames
        start_time = time.time()
        upscaled_frames = _upscale_frame_batch(frames, upsampler, scale)
        upscale_time = time.time() - start_time
        
        # Save upscaled video
        temp_output_path = Path(f"/tmp/{os.urandom(4).hex()}_upscaled.mp4")
        h, w = upscaled_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output_path), fourcc, 30.0, (w, h))
        for frame in upscaled_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        
        # Upload to S3
        output_url = await storage_client.upload_file(str(temp_output_path))
        temp_output_path.unlink()
        
        # Schedule deletion
        try:
            schedule_file_deletion(output_url.split('/')[-1].split('?')[0], 604800)
        except Exception as e:
            logger.warning(f"Failed to schedule deletion: {e}")
        
        # Final memory cleanup
        cleanup_memory()
        
        return {
            "status": "success",
            "output_url": output_url,
            "processing_time": upscale_time,
            "frames_processed": len(upscaled_frames),
            "scale": scale,
            "gpu_used": upsampler is not None
        }
        
    except Exception as e:
        logger.error(f"Upscale failed: {e}\n{traceback.format_exc()}")
        cleanup_memory()  # Ensure cleanup even on error
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}")

@app.get("/health")
async def health_check():
    gpu_memory_gb = get_gpu_memory_usage() / 1024
    return {
        "status": "ok",
        "gpu_memory_gb": round(gpu_memory_gb, 2),
        "memory_threshold_gb": MEMORY_THRESHOLD_GB,
        "models_cached": list(_model_cache.keys())
    }

@app.get("/memory-stats")
async def memory_stats():
    """Get detailed memory statistics."""
    gpu_memory_mb = get_gpu_memory_usage()
    
    # System memory
    system_mem = psutil.virtual_memory()
    
    return {
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_memory_gb": round(gpu_memory_mb / 1024, 2),
        "system_memory_total_gb": round(system_mem.total / (1024**3), 2),
        "system_memory_used_gb": round(system_mem.used / (1024**3), 2),
        "system_memory_percent": system_mem.percent,
        "memory_threshold_gb": MEMORY_THRESHOLD_GB,
        "gpu_under_threshold": gpu_memory_mb / 1024 <= MEMORY_THRESHOLD_GB
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting robust upscaling service with HD24K support...")
    uvicorn.run(app, host="0.0.0.0", port=29115)