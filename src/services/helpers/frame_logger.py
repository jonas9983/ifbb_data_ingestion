import json

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
        """
        frame_data = {
            # === METADATA ===
            'frame_number': frame_number,
            'frame_name': frame_name,
            'timestamp': datetime.now().isoformat(),
            'frame_width': frame_dims[1],
            'frame_height': frame_dims[0],
            
            # === SCENE DETECTION ===
            'is_camera_cut': is_camera_cut,
            'avg_pixel_difference': float(avg_pixel_diff),
            
            # === RAW FACE DETECTIONS ===
            'raw_faces': [
                {
                    'name': f['name'],
                    'bbox': f['bbox'].tolist() if isinstance(f['bbox'], np.ndarray) else f['bbox'],
                    'center_x': float(f['center_x']),
                    'confidence_score': float(f['score'])
                }
                for f in raw_faces
            ],
            
            # === TRACKED PERSONS & ASSOCIATIONS ===
            'tracked_athletes': [
                {
                    # Identity
                    'assigned_name': a.name,
                    'track_id': int(a.track_id) if a.track_id is not None else None,
                    
                    # Person Detection
                    'person_bbox': a.person_bbox,
                    'person_center_x': float(a.center_x),
                    'person_center_y': float((a.person_bbox[1] + a.person_bbox[3]) / 2),
                    'person_width': a.person_bbox[2] - a.person_bbox[0],
                    'person_height': a.person_bbox[3] - a.person_bbox[1],
                    'person_confidence': float(a.confidence),
                    
                    # Mask Data (store as RLE or polygon for space efficiency)
                    'has_mask': a.mask is not None,
                    'mask_area': int(np.sum(a.mask > 0)) if a.mask is not None else 0,
                    
                    # Face Association
                    'face_bbox': a.face_bbox if a.face_bbox is not None else None,
                    'face_detected': a.face_bbox is not None,
                    'face_name_raw': a.debug_raw_face_name if hasattr(a, 'debug_raw_face_name') else None,
                    'face_score': float(a.debug_face_score) if hasattr(a, 'debug_face_score') else 0.0,
                    
                    # Depth Classification
                    'is_front_row': a.is_front_row,
                    
                    # Marshall Detection
                    'is_marshall': a.name == "MARSHALL",
                    'marshall_score': float(a.debug_marshall_score) if hasattr(a, 'debug_marshall_score') else 0.0,
                }
                for a in athletes
            ],
            
            # === REGISTRY STATE (Track Memory) ===
            'registry_state': {
                str(track_id): name 
                for track_id, name in registry_state.items()
            },
            'marshall_track_id': int(marshall_track_id) if marshall_track_id != -1 else None,
            
            # === SWAP DETECTION ===
            'swap_detection': {
                'swaps_detected': swap_info.get('swaps', []),
                'ordered_lineup': swap_info.get('ordered_athletes', []),
                'lineup_positions': {
                    name: float(pos)
                    for name, pos in swap_info.get('positions', {}).items()
                }
            },
            
            # === STATISTICS ===
            'stats': {
                'total_persons_detected': len(athletes),
                'named_athletes': len([a for a in athletes if a.name != "Unknown" and a.name != "MARSHALL"]),
                'unknown_count': len([a for a in athletes if a.name == "Unknown"]),
                'marshall_detected': any(a.name == "MARSHALL" for a in athletes),
                'front_row_count': len([a for a in athletes if a.is_front_row]),
                'back_row_count': len([a for a in athletes if not a.is_front_row]),
                'total_faces_detected': len(raw_faces)
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
                'version': '2.0_comprehensive'
            },
            'frames': self.frame_logs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Comprehensive frame data exported to: {output_path}")
        print(f"   Total frames logged: {len(self.frame_logs)}")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
