# HAT-L Video Upscaling Service

This service utilizes the HAT-L (Hybrid Attention Transformer Large) model to perform high-quality video upscaling. HAT-L achieves significantly higher perceptual quality scores on the PieAPP metric compared to Real-ESRGAN, directly improving miner rewards on this subnet. The service is primarily designed for HD-to-4K video upscaling, supporting resolutions such as 1080x2048 to 2160x4096, with both 2X and 4X upscaling capabilities.

## Features
- **State-of-the-Art Upscaling**: Uses HAT-L (Hybrid Attention Transformer Large) model via the spandrel library for superior perceptual quality.
- **Higher PieAPP Scores**: HAT-L achieves significantly better perceptual quality scores compared to Real-ESRGAN, directly increasing miner emissions.
- **Frame Loss Prevention**: The service addresses frame loss issues by duplicating the last frame at the end of the video before processing, ensuring frame consistency.
- **GPU Acceleration**: Optimized for NVIDIA RTX 4090 hardware with CUDA support.

## Architecture

The service uses:
- **spandrel**: Modern Python library for loading and running super-resolution models
- **HAT-L**: Hybrid Attention Transformer Large model with 24 blocks, 180 embedding dimensions, and 6 attention heads
- **PyTorch**: For efficient GPU-accelerated inference
- **OpenCV**: For frame processing and image I/O

## Installation

The required dependencies are automatically installed via pip:

```bash
pip install spandrel==0.4.1
```

Ensure you have the following system dependencies:
- Python 3.8+
- CUDA-capable GPU (NVIDIA RTX 4090 recommended)
- FFmpeg (for video encoding/decoding)

## Usage

The service runs as a FastAPI server that accepts upscaling requests:

```python
# Start the server
python services/upscaling/server.py
```

### API Endpoint

**POST** `/upscale-video`

Request body:

{
    "payload_url": "https://example.com/video.mp4",
    "task_type": "SD24K"  // or "upscale" for 2X
}
```

Response:

{
    "uploaded_video_url": "https://storage.example.com/upscaled_video.mp4"
}
```

## Technical Details

### HAT-L Model Configuration
- **Upscaling factors**: 2X and 4X
- **Architecture**: Hybrid Attention Transformer
- **Parameters**: ~20M (optimized for RTX 4090)
- **Input**: Video frames extracted via FFmpeg
- **Output**: Upscaled frames encoded with libx265

### Processing Pipeline
1. Download input video from provided URL
2. Duplicate last 2 frames to prevent frame loss
3. Extract frames using FFmpeg
4. Process each frame with HAT-L model (GPU-accelerated)
5. Encode upscaled frames back to video with libx265
6. Upload result to storage and return presigned URL

## Performance

- **Target Hardware**: NVIDIA RTX 4090
- **Expected Performance**: ~1-2 seconds per frame at 1080p input
- **Memory Usage**: ~8-12GB VRAM depending on input resolution

## Known Issues
- **Frame Loss**: The service automatically handles frame loss by duplicating the last frame before processing.
- **Performance**: Large videos require substantial computational power. Ensure your system meets the requirements.

---
**Note**: This service is optimized for Ubuntu 24.04 with NVIDIA RTX 4090 GPUs. Compatibility with other configurations may vary.
