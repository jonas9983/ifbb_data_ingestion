import os
import json
from typing import List, Dict, Any

class FrameLogger:
    """
    Logs structured data optimized for the React visualization app.
    """
    def __init__(self):
        self.frame_logs = []
        
    def log_frame(
        self,
        frame_number: int,
        athletes: List[Any],
        is_camera_cut: bool,
        swap_info: Dict
    ):
        # 1. Format swaps
        swaps = swap_info.get('swaps', [])
        formatted_swaps = [f"{s[0]} <-> {s[1]}" for s in swaps]

        # 2. Build the clean JSON payload
        frame_data = {
            'frame': int(frame_number),
            'is_camera_cut': bool(is_camera_cut),
            'lineup': [str(name) for name in swap_info.get('ordered_athletes', [])],
            
            'marshall': {
                'detected': False,
                'center_x': None
            },
            
            'swaps_detected': formatted_swaps
        }
        
        self.frame_logs.append(frame_data)
    
    def export_to_json(self, output_path: str):
        """
        Export all logged frames to a lightweight JSON file.
        """
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_logs, f, indent=2, ensure_ascii=False)
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n UI JSON exported: {output_path}")
        print(f"   Total frames logged: {len(self.frame_logs)}")
        print(f"   File size: {file_size_mb:.4f} MB")