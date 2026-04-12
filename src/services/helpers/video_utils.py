import cv2
from typing import Tuple, Optional

def parse_frame_range(range_str: Optional[str]) -> Tuple[int, int, int]:
    """Parse frame range string. If None, returns flags to auto-detect all frames."""
    if not range_str: 
        return 1, -1, 1  # -1 signals we need to read to the end of the video
        
    parts = range_str.split(':')
    start = int(parts[0]) if len(parts) > 0 and parts[0] else 1
    end = int(parts[1]) if len(parts) > 1 and parts[1] else -1
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return start, end, step

def get_video_frames(video_path: str, start_f: int, end_f: int, step: int = 1):
    """Reads frames directly from a local mp4 video file."""
    if not video_path or not cv2.os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_end_f = total_frames if end_f == -1 else min(end_f, total_frames)
    
    print(f"\n Reading Video: {total_frames} total frames available")
    
    # Fast-forward to the starting frame (OpenCV frames are 0-indexed)
    if start_f > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f - 1)
    
    frame_idx = start_f

    while frame_idx <= actual_end_f:
        ret, img = cap.read()
        if not ret:
            break
        
        # Only process if it matches our step interval
        if (frame_idx - start_f) % step == 0:
            frame_name = f"frame_{frame_idx:05d}.jpg"
            yield frame_idx, frame_name, img
            
        frame_idx += 1

    cap.release()