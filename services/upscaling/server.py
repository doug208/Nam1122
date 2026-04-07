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
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple
import threading

app = FastAPI()

# Global Real-ESRGAN model instance (loaded once at startup)
realesrgan_model = None
realesrgan_device = None


class RealESRGANUpscaler:
    """Real-ESRGAN upscaler optimized for RTX 4090 with 24GB VRAM."""
    
    # Official Real-ESRGAN model URLs
    MODEL_URLS = {
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    }
    
    def __init__(self, model_name: str = "RealESRGAN_x4plus", tile_size: int = 2048):
        """
        Initialize Real-ESRGAN model.
        
        Args:
            model_name: Name of the Real-ESRGAN model to use
            tile_size: Tile size for processing (optimized for 24GB VRAM - 2048 for full utilization)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for upscaling.")
        
        # Optimized for RTX 4090 24GB VRAM with FP16
        # 2048 tile size allows processing 4K frames efficiently
        self.tile_size = tile_size
        self.tile_pad = 32
        self.model_name = model_name
        self.model = None
        self.scale = 4 if "x4" in model_name else 2
        
        self._load_model()
    
    def _download_model_weights(self, model_path: str) -> str:
        """Download Real-ESRGAN model weights from official source."""
        import urllib.request
        import ssl
        
        if self.model_name not in self.MODEL_URLS:
            raise RuntimeError(f"Unknown model: {self.model_name}. Available: {list(self.MODEL_URLS.keys())}")
        
        url = self.MODEL_URLS[self.model_name]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info(f"Downloading {self.model_name} weights from {url}...")
        
        # Create SSL context that allows us to download from GitHub
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            if block_num % 100 == 0:  # Log every 100 blocks
                logger.info(f"Download progress: {percent:.1f}%")
        
        try:
            urllib.request.urlretrieve(url, model_path, reporthook=download_progress)
            logger.info(f"Model weights downloaded to {model_path}")
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            raise RuntimeError(f"Failed to download model weights: {e}")
        
        return model_path
    
    def _validate_weights_loaded(self):
        """Validate that model weights were loaded correctly by running a test inference."""
        import torch.nn.functional as F
        
        # Create a small test input
        test_input = torch.randn(1, 3, 64, 64, device=self.device, dtype=torch.float16)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                test_output = self.model(test_input)
        
        # Check for NaN or Inf values which indicate bad weights
        if torch.isnan(test_output).any() or torch.isinf(test_output).any():
            raise RuntimeError("Model weights validation failed: output contains NaN or Inf values")
        
        # Check that output has expected scale
        expected_size = 64 * self.scale
        if test_output.shape[2] != expected_size or test_output.shape[3] != expected_size:
            raise RuntimeError(f"Model output size mismatch: expected {expected_size}x{expected_size}, got {test_output.shape[2]}x{test_output.shape[3]}")
        
        logger.info("Model weights validation passed")
        torch.cuda.empty_cache()
    
    def _load_model(self):
        """Load Real-ESRGAN model with FP16 optimization and automatic weight downloading."""
        try:
            # Import basicsr arch for Real-ESRGAN
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Initialize model architecture
            if self.model_name == "RealESRGAN_x4plus":
                self.model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                self.scale = 4
            elif self.model_name == "RealESRGAN_x2plus":
                self.model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2
                )
                self.scale = 2
            else:
                raise RuntimeError(f"Unsupported model: {self.model_name}. Use RealESRGAN_x4plus or RealESRGAN_x2plus")
            
            # Get model path and download if needed
            model_path = self._get_model_path()
            
            if not os.path.exists(model_path):
                logger.warning(f"Model weights not found at {model_path}, attempting download...")
                model_path = self._download_model_weights(model_path)
            
            # Load pre-trained weights
            logger.info(f"Loading model weights from {model_path}...")
            load_net = torch.load(model_path, map_location=self.device)
            
            # Real-ESRGAN checkpoint stores weights as nested dict under 'params_ema' or 'params'
            if 'params_ema' in load_net:
                load_net = load_net['params_ema']
            elif 'params' in load_net:
                load_net = load_net['params']
            
            self.model.load_state_dict(load_net, strict=True)
            logger.info(f"Model weights loaded successfully from {model_path}")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Enable FP16 for faster inference on RTX 4090
            self.model = self.model.half()
            
            # Validate weights were loaded correctly
            self._validate_weights_loaded()
            
            logger.info(f"Real-ESRGAN model loaded on {self.device} with FP16 enabled")
            logger.info(f"Tile size: {self.tile_size}, Scale: {self.scale}")
            
            # Log GPU memory info
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU memory: {allocated:.2f}GB allocated / {total_memory:.2f}GB total")
            
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            raise RuntimeError(f"Failed to load Real-ESRGAN model: {e}")
    
    def _get_model_path(self) -> str:
        """Get path to model weights file."""
        # Use a consistent path in the service directory
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir / f"{self.model_name}.pth")
    
    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process image for model input."""
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        img = img.half()  # Convert to FP16
        return img
    
    def post_process(self, output: torch.Tensor) -> np.ndarray:
        """Post-process model output to image."""
        output = output.float().cpu().squeeze(0)
        output = output[[2, 1, 0], :, :].numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        return output
    
    @torch.no_grad()
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale image using tiled processing.
        
        Args:
            img: Input image (BGR format from OpenCV)
            
        Returns:
            Upscaled image
        """
        img_tensor = self.pre_process(img)
        
        # Check if image needs tiling
        _, _, h, w = img_tensor.size()
        
        if h <= self.tile_size and w <= self.tile_size:
            # Small image, process directly
            with torch.cuda.amp.autocast():
                output = self.model(img_tensor)
        else:
            # Large image, use tiled processing
            output = self._tiled_upscale(img_tensor)
        
        return self.post_process(output)
    
    def upscale_batch(self, frames: list, batch_size: int = 4) -> list:
        """
        Upscale multiple frames in batches for better GPU utilization.
        
        Args:
            frames: List of input images (BGR format from OpenCV)
            batch_size: Number of frames to process in parallel
            
        Returns:
            List of upscaled images
        """
        upscaled_frames = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_tensors = []
            
            # Pre-process all frames in batch
            for frame in batch:
                img_tensor = self.pre_process(frame)
                batch_tensors.append(img_tensor)
            
            # Stack into batch tensor
            if len(batch_tensors) > 1:
                batch_input = torch.cat(batch_tensors, dim=0)
            else:
                batch_input = batch_tensors[0]
            
            # Process batch
            with torch.cuda.amp.autocast():
                _, _, h, w = batch_input.size()
                
                if h <= self.tile_size and w <= self.tile_size:
                    # Small images, process directly
                    batch_output = self.model(batch_input)
                else:
                    # Large images, process each frame with tiling
                    outputs = []
                    for j in range(batch_input.size(0)):
                        single_input = batch_input[j:j+1]
                        single_output = self._tiled_upscale(single_input)
                        outputs.append(single_output)
                    batch_output = torch.cat(outputs, dim=0)
            
            # Post-process outputs
            for j in range(batch_output.size(0)):
                output = batch_output[j:j+1]
                upscaled_frame = self.post_process(output)
                upscaled_frames.append(upscaled_frame)
            
            # Clear GPU cache after each batch to prevent memory accumulation
            if (i // batch_size) % 2 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        return upscaled_frames
    
    def _tiled_upscale(self, img: torch.Tensor) -> torch.Tensor:
        """
        Process large images using tiled processing.
        Optimized for RTX 4090 24GB VRAM.
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        
        # Initialize output tensor
        output = img.new_zeros(output_shape)
        
        # Calculate tiles
        tiles_x = (width + self.tile_size - 1) // self.tile_size
        tiles_y = (height + self.tile_size - 1) // self.tile_size
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile coordinates
                start_y = y * self.tile_size
                start_x = x * self.tile_size
                end_y = min(start_y + self.tile_size, height)
                end_x = min(start_x + self.tile_size, width)
                
                # Add padding for seamless blending
                pad_top = self.tile_pad if y > 0 else 0
                pad_bottom = self.tile_pad if y < tiles_y - 1 else 0
                pad_left = self.tile_pad if x > 0 else 0
                pad_right = self.tile_pad if x < tiles_x - 1 else 0
                
                tile_start_y = max(0, start_y - pad_top)
                tile_end_y = min(height, end_y + pad_bottom)
                tile_start_x = max(0, start_x - pad_left)
                tile_end_x = min(width, end_x + pad_right)
                
                # Extract tile
                tile = img[:, :, tile_start_y:tile_end_y, tile_start_x:tile_end_x]
                
                # Process tile
                with torch.cuda.amp.autocast():
                    tile_output = self.model(tile)
                
                # Calculate output coordinates
                out_start_y = start_y * self.scale
                out_end_y = end_y * self.scale
                out_start_x = start_x * self.scale
                out_end_x = end_x * self.scale
                
                # Calculate valid region (remove padding from output)
                valid_start_y = pad_top * self.scale
                valid_end_y = tile_output.size(2) - (pad_bottom * self.scale)
                valid_start_x = pad_left * self.scale
                valid_end_x = tile_output.size(3) - (pad_right * self.scale)
                
                # Place tile in output
                output[:, :, out_start_y:out_end_y, out_start_x:out_end_x] = \
                    tile_output[:, :, valid_start_y:valid_end_y, valid_start_x:valid_end_x]
        
        return output


def initialize_upscaler():
    """Initialize the global Real-ESRGAN upscaler at startup."""
    global realesrgan_model, realesrgan_device
    
    try:
        # Use tile size optimized for RTX 4090 24GB VRAM
        # 2048 tile size allows efficient processing of 4K frames
        # With FP16, this utilizes ~16-20GB of VRAM effectively
        realesrgan_model = RealESRGANUpscaler(
            model_name="RealESRGAN_x4plus",
            tile_size=2048
        )
        realesrgan_device = realesrgan_model.device
        logger.info("Real-ESRGAN upscaler initialized successfully at startup")
    except Exception as e:
        logger.error(f"Failed to initialize Real-ESRGAN upscaler: {e}")
        raise RuntimeError(f"Failed to initialize Real-ESRGAN upscaler: {e}")


# Initialize model at module load (startup)
initialize_upscaler()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    # output_file_upscaled: Optional[str] = None
    

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
    Upscales a video using Real-ESRGAN Python API optimized for RTX 4090 with streaming pipeline.
    
    Uses the globally loaded model with CUDA FP16 inference and tiled processing.
    Frames are streamed directly from ffmpeg decoder to upscaler to ffmpeg encoder without disk I/O.
    Raises error on GPU failure - no fallback to CPU methods.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform.

    Returns:
        str: The full path to the upscaled video.
    """
    global realesrgan_model
    
    try:
        input_file = Path(payload_video_path)
        
        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")
        
        # Check if model is loaded
        if realesrgan_model is None:
            raise RuntimeError("Real-ESRGAN model not initialized. GPU upscaling unavailable.")
        
        # Determine scale factor based on task type
        if task_type == "SD24K":
            target_scale = 4
        else:
            target_scale = 2
        
        # Get video properties
        frame_rate = get_frame_rate(input_file)
        logger.info(f"Frame rate detected: {frame_rate} fps")
        
        # Generate output file path
        output_file_upscaled = input_file.with_name(f"{input_file.stem}_upscaled.mp4")
        
        # Create streaming pipeline: ffmpeg decode -> upscaler -> ffmpeg encode
        logger.info("Starting streaming video upscaling pipeline...")
        start_time = time.time()
        
        # Start ffmpeg decoding process (raw video frames to stdout)
        decode_command = [
            "ffmpeg",
            "-i", str(input_file),
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ]
        
        decode_process = subprocess.Popen(
            decode_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Start ffmpeg encoding process (raw video frames from stdin)
        encode_command = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{realesrgan_model.tile}x{realesrgan_model.tile}",
            "-r", str(frame_rate),
            "-i", "-",
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
            str(output_file_upscaled)
        ]
        
        encode_process = subprocess.Popen(
            encode_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Process frames in streaming fashion
        frame_count = 0
        while True:
            # Read raw frame data from decoder
            raw_frame_data = decode_process.stdout.read(realesrgan_model.tile * realesrgan_model.tile * 3)
            if not raw_frame_data:
                break
            
            # Convert raw data to numpy array
            frame_array = np.frombuffer(raw_frame_data, dtype=np.uint8)
            frame_array = frame_array.reshape((realesrgan_model.tile, realesrgan_model.tile, 3))
            
            # Upscale frame
            upscaled_frame = realesrgan_model.upscale(frame_array)
            
            # If target scale differs from model scale, resize accordingly
            if target_scale != realesrgan_model.scale:
                h, w = upscaled_frame.shape[:2]
                new_h = int(h * target_scale / realesrgan_model.scale)
                new_w = int(w * target_scale / realesrgan_model.scale)
                upscaled_frame = cv2.resize(upscaled_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Write upscaled frame to encoder
            encode_process.stdin.write(upscaled_frame.tobytes())
            
            frame_count += 1
            
            # Periodic GPU memory cleanup
            if frame_count % 20 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        # Close processes
        decode_process.stdout.close()
        decode_process.wait()
        
        encode_process.stdin.close()
        encode_process.wait()
        
        # Check for errors
        if decode_process.returncode != 0:
            error_output = decode_process.stderr.read().decode()
            raise HTTPException(status_code=500, detail=f"Frame decoding failed: {error_output}")
        
        if encode_process.returncode != 0:
            error_output = encode_process.stderr.read().decode()
            raise HTTPException(status_code=500, detail=f"Video encoding failed: {error_output}")
        
        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled video file was not created")
        
        logger.info(f"Streaming upscaling completed in {time.time() - start_time:.2f} seconds for {frame_count} frames")
        
        # Cleanup input file
        if input_file.exists():
            input_file.unlink()
        
        logger.info(f"Upscaled video saved to: {output_file_upscaled}")
        return str(output_file_upscaled)
        
    except HTTPException:
        raise
    except RuntimeError as e:
        # GPU failure - raise error, do not fall back
        logger.error(f"GPU upscaling failed: {e}")
        raise HTTPException(status_code=500, detail=f"GPU upscaling failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during upscaling: {e}")
        raise HTTPException(status_code=500, detail=f"Upscaling failed: {e}")

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