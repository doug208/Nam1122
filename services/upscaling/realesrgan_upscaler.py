import sys
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import time

def upscale_video_realesrgan(input_video_path: str, output_video_path: str, scale: int = 4, frame_rate: float = 30.0):
    """
    Upscales a video using Real-ESRGAN with CUDA GPU acceleration and batch processing.
    
    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the upscaled video
        scale (int): Upscaling factor (2 or 4)
        frame_rate (float): Frame rate of the input video
    """
    print(f"Starting Real-ESRGAN upscaling with scale factor {scale}x")
    
    # Initialize Real-ESRGAN model
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
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=False,  # Use FP32 for better compatibility
        device=device
    )
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {input_video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate output dimensions
    output_width = width * scale
    output_height = height * scale
    
    print(f"Input video: {width}x{height}, {total_frames} frames")
    print(f"Output video: {output_width}x{output_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (output_width, output_height))
    
    if not out.isOpened():
        raise Exception(f"Cannot create output video file: {output_video_path}")
    
    # Process frames in batches for GPU efficiency
    batch_size = 4  # Adjust based on GPU memory
    frame_count = 0
    batch_frames = []
    batch_indices = []
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames in the last batch
            if batch_frames:
                _process_batch(batch_frames, batch_indices, upsampler, out)
            break
        
        # Add frame to batch
        batch_frames.append(frame)
        batch_indices.append(frame_count)
        frame_count += 1
        
        # Process batch when full
        if len(batch_frames) >= batch_size:
            _process_batch(batch_frames, batch_indices, upsampler, out)
            batch_frames = []
            batch_indices = []
        
        # Print progress
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed {frame_count}/{total_frames} frames ({fps:.2f} FPS)")
    
    # Clean up
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"Upscaling completed in {total_time:.2f} seconds")
    print(f"Average processing speed: {frame_count/total_time:.2f} FPS")


def _process_batch(frames, indices, upsampler, video_writer):
    """
    Process a batch of frames using Real-ESRGAN.
    """
    for i, frame in enumerate(frames):
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Upscale frame
            output, _ = upsampler.enhance(frame_rgb, outscale=upsampler.scale)
            
            # Convert RGB back to BGR
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Write to video
            video_writer.write(output_bgr)
            
        except Exception as e:
            print(f"Error processing frame {indices[i]}: {e}")
            # Write original frame if enhancement fails
            original_upscaled = cv2.resize(frame, None, fx=upsampler.scale, fy=upsampler.scale, interpolation=cv2.INTER_CUBIC)
            video_writer.write(original_upscaled)