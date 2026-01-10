import os
import json
from typing import Tuple, List, Dict, Any
from datetime import datetime
import numpy as np

def numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    else:
        return obj


class FrameLogger:
    """
    Logs all frame data in a structured format for post-processing.
    """
    def __init__(self):
        self.frame_logs = []
        
    def log_frame(
        self,
        frame_number: int,
        frame_name: str,
        athletes: List[Any],
        raw_faces: List[Dict],
        is_camera_cut: bool,
        avg_pixel_diff: float,
        registry_state: Dict[int, str],
        marshall_track_id: int,
        swap_info: Dict,
        frame_dims: Tuple[int, int]
    ):
        """
        Log comprehensive data for a single frame.
        All numpy arrays are converted to lists for JSON serialization.
        """
        frame_data = {
            # === METADATA ===
            'frame_number': int(frame_number),
            'frame_name': str(frame_name),
            'timestamp': datetime.now().isoformat(),
            'frame_width': int(frame_dims[1]),
            'frame_height': int(frame_dims[0]),
            
            # === SCENE DETECTION ===
            'is_camera_cut': bool(is_camera_cut),
            'avg_pixel_difference': float(avg_pixel_diff),
            
            # === RAW FACE DETECTIONS ===
            'raw_faces': [
                {
                    'name': str(f['name']),
                    'bbox': numpy_to_python(f['bbox']),
                    'center_x': float(f['center_x']),
                    'confidence_score': float(f['score'])
                }
                for f in raw_faces
            ],
            
            # === TRACKED PERSONS & ASSOCIATIONS ===
            'tracked_athletes': [
                {
                    # Identity
                    'assigned_name': str(a.name),
                    'track_id': int(a.track_id) if a.track_id is not None else None,
                    
                    # Person Detection
                    'person_bbox': numpy_to_python(a.person_bbox),
                    'person_center_x': float(a.center_x),
                    'person_center_y': float((a.person_bbox[1] + a.person_bbox[3]) / 2),
                    'person_width': int(a.person_bbox[2] - a.person_bbox[0]),
                    'person_height': int(a.person_bbox[3] - a.person_bbox[1]),
                    'person_confidence': float(a.confidence),
                    
                    # Mask Data
                    'has_mask': bool(a.mask is not None),
                    'mask_area': int(np.sum(a.mask > 0)) if a.mask is not None else 0,
                    
                    # Face Association
                    'face_bbox': numpy_to_python(a.face_bbox) if a.face_bbox is not None else None,
                    'face_detected': bool(a.face_bbox is not None),
                    'face_name_raw': str(a.debug_raw_face_name) if hasattr(a, 'debug_raw_face_name') else None,
                    'face_score': float(a.debug_face_score) if hasattr(a, 'debug_face_score') else 0.0,
                    
                    # Depth Classification
                    'is_front_row': bool(a.is_front_row),
                    
                    # Marshall Detection
                    'is_marshall': bool(a.name == "MARSHALL"),
                    'marshall_score': float(a.debug_marshall_score) if hasattr(a, 'debug_marshall_score') else 0.0,
                }
                for a in athletes
            ],
            
            # === REGISTRY STATE (Track Memory) ===
            'registry_state': {
                str(track_id): str(name) 
                for track_id, name in registry_state.items()
            },
            'marshall_track_id': int(marshall_track_id) if marshall_track_id != -1 else None,
            
            # === SWAP DETECTION ===
            'swap_detection': {
                'swaps_detected': [
                    {'athlete_1': str(a), 'athlete_2': str(b)} 
                    for a, b in swap_info.get('swaps', [])
                ],
                'ordered_lineup': [str(name) for name in swap_info.get('ordered_athletes', [])],
                'lineup_positions': {
                    str(name): float(pos)
                    for name, pos in swap_info.get('positions', {}).items()
                }
            },
            
            # === STATISTICS ===
            'stats': {
                'total_persons_detected': int(len(athletes)),
                'named_athletes': int(len([a for a in athletes if a.name != "Unknown" and a.name != "MARSHALL"])),
                'unknown_count': int(len([a for a in athletes if a.name == "Unknown"])),
                'marshall_detected': bool(any(a.name == "MARSHALL" for a in athletes)),
                'front_row_count': int(len([a for a in athletes if a.is_front_row])),
                'back_row_count': int(len([a for a in athletes if not a.is_front_row])),
                'total_faces_detected': int(len(raw_faces))
            }
        }
        
        self.frame_logs.append(frame_data)
    
    def export_to_json(self, output_path: str):
        """
        Export all logged frames to a JSON file.
        """
        output_data = {
            'metadata': {
                'total_frames': len(self.frame_logs),
                'export_timestamp': datetime.now().isoformat(),
                'version': '2.1_fixed_serialization'
            },
            'frames': self.frame_logs
        }
        
        # Use numpy_to_python as a safety net
        output_data = numpy_to_python(output_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n✅ Comprehensive frame data exported to: {output_path}")
        print(f"   Total frames logged: {len(self.frame_logs)}")
        print(f"   File size: {file_size_mb:.2f} MB")
