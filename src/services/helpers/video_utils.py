import os
import glob
import cv2
import requests
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

def parse_frame_range(range_str: Optional[str]) -> Tuple[int, int, int]:
    """Parse frame range string. If None, returns flags to auto-detect all frames."""
    if not range_str: 
        return 1, -1, 1  # -1 signals we need to auto-detect the end
        
    parts = range_str.split(':')
    start = int(parts[0]) if len(parts) > 0 and parts[0] else 1
    end = int(parts[1]) if len(parts) > 1 and parts[1] else -1
    step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return start, end, step

def _find_max_frame_on_server(base_url: str, start_f: int) -> int:
    """Find the last valid frame on the server without downloading them."""
    print(" Probing server to find the total number of frames...")
    
    upper = start_f
    while requests.head(base_url.format(upper), timeout=5).status_code == 200:
        upper += 500
        
    low = max(start_f, upper - 500)
    high = upper
    
    while low < high:
        mid = (low + high) // 2
        if requests.head(base_url.format(mid), timeout=5).status_code == 200:
            low = mid + 1
        else:
            high = mid
            
    print(f" Found last frame at: {low - 1}")
    return low - 1

def download_frames_parallel(base_url: str, save_dir: str, start_f: int, end_f: int, step: int = 1):
    """Downloads frames heavily in parallel."""
    os.makedirs(save_dir, exist_ok=True)
    
    if end_f == -1:
        end_f = _find_max_frame_on_server(base_url, start_f)
        
    indices = list(range(start_f, end_f + 1, step))
    print(f"\n--- Checking/Downloading {len(indices)} frames to {save_dir} ---")
    
    def fetch(idx):
        filename = f"shots_{idx:05d}.png"
        filepath = os.path.join(save_dir, filename)
        
        if os.path.exists(filepath): return True
            
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

    with ThreadPoolExecutor(max_workers=50) as executor:
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
    
    actual_end_f = float('inf') if end_f == -1 else end_f
    
    for path in image_paths:
        try:
            frame_number = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
        except ValueError:
            continue

        if frame_number < start_f or frame_number > actual_end_f or frame_number % step != 0: 
            continue

        img = cv2.imread(path)
        if img is not None:
            yield frame_number, os.path.basename(path), img