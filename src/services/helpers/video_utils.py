import os
import glob
import cv2
import requests
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

def parse_frame_range(range_str: str) -> Tuple[int, int, int]:
    """Parse frame range string in format 'start:end:step'."""
    if not range_str: return 1, 100, 1
    parts = range_str.split(':')
    start = int(parts[0]) if len(parts) > 0 and parts[0] else 1
    end = int(parts[1]) if len(parts) > 1 and parts[1] else 100
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return start, end, step

def download_frames_parallel(base_url: str, save_dir: str, start_f: int, end_f: int, step: int = 1):
    """Downloads frames heavily in parallel before tracking begins. Skips existing files."""
    os.makedirs(save_dir, exist_ok=True)
    indices = list(range(start_f, end_f + 1, step))
    
    print(f"\n--- Checking/Downloading {len(indices)} frames to {save_dir} ---")
    
    def fetch(idx):
        filename = f"shots_{idx:05d}.png"
        filepath = os.path.join(save_dir, filename)
        
        if os.path.exists(filepath):
            return True
            
        url = base_url.format(idx)
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                return True
        except Exception:
            pass
        return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch, indices))
        
    success_count = sum(results)
    if success_count < len(indices):
        print(f" Warning: Only downloaded {success_count}/{len(indices)} frames.")
    else:
        print(f" All {success_count} frames ready locally!")

def get_local_images(input_dir: str, start_f: int, end_f: int, step: int = 1):
    """Reads images directly from the local drive."""
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")) + 
                         glob.glob(os.path.join(input_dir, "*.jpg")))
    
    for path in image_paths:
        try:
            frame_number = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
        except ValueError:
            continue

        if frame_number < start_f or frame_number > end_f or frame_number % step != 0: 
            continue

        img = cv2.imread(path)
        if img is not None:
            yield frame_number, os.path.basename(path), img