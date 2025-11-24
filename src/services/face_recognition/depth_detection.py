import numpy as np
from typing import List, Tuple, Any
from sklearn.cluster import KMeans

class DepthAnalyzer:
    """
    Determines if athletes are in the front row or back row using
    Y-Max (Feet) clustering and Segmentation Mask Containment checks.
    """
    
    def __init__(self):
        # If > 40% of a person's mask is covered by someone else, they are in back.
        self.mask_overlap_threshold = 0.4 
        
        # Fallback for bounding boxes if masks fail
        self.bbox_overlap_threshold = 0.6 

    def _get_feet_y(self, athlete: Any) -> int:
        """Returns the Y-coordinate of the athlete's feet (bottom of bbox)."""
        bbox = athlete.person_bbox if athlete.person_bbox else athlete.face_bbox
        return bbox[3] # y2

    def _get_bbox(self, athlete: Any) -> List[int]:
        return athlete.person_bbox if athlete.person_bbox else athlete.face_bbox

    def _is_behind(self, inner_athlete: Any, outer_athlete: Any) -> bool:
        """
        Check if inner_athlete is physically behind outer_athlete.
        Priority: Segmentation Masks. Fallback: Bounding Boxes.
        """
        # --- STRATEGY 1: MASK BASED CHECK (Pixel Perfect) ---
        if inner_athlete.mask is not None and outer_athlete.mask is not None:
            mask_inner = (inner_athlete.mask > 0)
            mask_outer = (outer_athlete.mask > 0)
            
            # Calculate Intersection (pixels shared by both)
            intersection = np.logical_and(mask_inner, mask_outer).sum()
            inner_area = mask_inner.sum()
            
            if inner_area == 0:
                return False
                
            overlap_ratio = intersection / inner_area
            
            # If significant overlap, the smaller/inner one is likely behind
            return overlap_ratio > self.mask_overlap_threshold

        # --- STRATEGY 2: BBOX FALLBACK ---
        in_box = self._get_bbox(inner_athlete)
        out_box = self._get_bbox(outer_athlete)
        
        ix1 = max(in_box[0], out_box[0])
        iy1 = max(in_box[1], out_box[1])
        ix2 = min(in_box[2], out_box[2])
        iy2 = min(in_box[3], out_box[3])

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        inner_area = (in_box[2] - in_box[0]) * (in_box[3] - in_box[1])
        
        if inner_area > 0:
            return (intersection_area / inner_area) > self.bbox_overlap_threshold
            
        return False

    def filter_front_row_athletes(
        self,
        athletes: List[Any],
        frame_height: int,
        verbose: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        """
        Splits athletes into [front_row, back_row] based on:
        1. Physical Overlap (One person blocking another)
        2. Y-Axis Position (Feet position clustering)
        """
        if not athletes:
            return [], []
        
        # --- STEP 1: Overlap / Containment Check ---
        # Sort by Feet Position (Highest Y value first -> Closest to camera)
        # We want to check if the people "in back" are covered by people "in front"
        sorted_indices = np.argsort([-self._get_feet_y(a) for a in athletes])
        
        valid_front_indices = []
        rejected_by_overlap = []
        
        for i in sorted_indices:
            current_athlete = athletes[i]
            is_hidden = False
            
            # Check if this person is hidden behind anyone currently deemed "in front"
            for valid_idx in valid_front_indices:
                front_athlete = athletes[valid_idx]
                
                if self._is_behind(current_athlete, front_athlete):
                    is_hidden = True
                    if verbose:
                        print(f"  [Depth] Overlap detected: {current_athlete.name} is behind {front_athlete.name}")
                    break
            
            if is_hidden:
                rejected_by_overlap.append(current_athlete)
            else:
                valid_front_indices.append(i)

        # Candidates for front row after removing physically blocked people
        candidates = [athletes[i] for i in valid_front_indices]
        
        if len(candidates) < 2:
            # Not enough people to cluster, return what we have
            return candidates, rejected_by_overlap

        # --- STEP 2: Clustering based on Feet Position (Y-Max) ---
        y_max_values = np.array([self._get_feet_y(a) for a in candidates]).reshape(-1, 1)

        # Calculate spread of feet positions
        y_spread = np.max(y_max_values) - np.min(y_max_values)
        
        # If feet are within 5% of frame height, assume they are in one line (single row)
        if y_spread < (frame_height * 0.05):
            if verbose: print(f"  [Depth] Low Y-variance ({y_spread}px), assuming single row.")
            return candidates, rejected_by_overlap

        try:
            if verbose: print(f"  [Depth] High Y-variance ({y_spread}px), clustering...")
            
            # K-Means with 2 clusters (Front Row vs Back Row)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(y_max_values)
            
            # Identify which cluster is "Front"
            # Higher Y-pixel value = Lower on screen = Closer to camera
            center_0 = kmeans.cluster_centers_[0][0]
            center_1 = kmeans.cluster_centers_[1][0]
            
            front_label = 0 if center_0 > center_1 else 1
            
            final_front_row = []
            final_back_row = list(rejected_by_overlap) # Start with overlapped people
            
            for i, athlete in enumerate(candidates):
                if labels[i] == front_label:
                    final_front_row.append(athlete)
                else:
                    final_back_row.append(athlete)
                    if verbose:
                        print(f"  [Depth] Clustering moved {athlete.name} to back row (Y={self._get_feet_y(athlete)})")
                    
            return final_front_row, final_back_row

        except Exception as e:
            print(f"  [Depth] Clustering failed ({e}), defaulting to overlap check only.")
            return candidates, rejected_by_overlap