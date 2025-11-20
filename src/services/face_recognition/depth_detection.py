import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans

class DepthAnalyzer:
    """
    Determines if athletes are in the front row or back row using
    Y-Max (Feet) clustering and Mask Containment checks.
    """
    
    def __init__(self):
        # Mask overlap is usually very low for two distinct people (even if one is behind)
        # BBox overlap is usually high. We lower the threshold for masks.
        self.containment_threshold = 0.6 
        self.mask_overlap_threshold = 0.4 # If > 40% of pixels overlap, it's likely the same person

    def _get_bbox_coords(self, athlete):
        """Helper to safely get bbox [x1, y1, x2, y2]."""
        # Prefer person body box, fallback to face
        bbox = athlete.person_bbox if athlete.person_bbox else athlete.face_bbox
        return bbox

    def calculate_bbox_area(self, bbox: List[int]) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _is_contained(self, inner_athlete, outer_athlete) -> bool:
        """
        Check if inner_athlete is significantly overlapping/inside outer_athlete.
        PRIORITY: Uses Segmentation Masks (Pixel Perfect).
        FALLBACK: Uses Bounding Boxes (Rectangle Approximation).
        """
        # --- STRATEGY 1: MASK BASED CHECK (ACCURATE) ---
        if inner_athlete.mask is not None and outer_athlete.mask is not None:
            mask1 = inner_athlete.mask
            mask2 = outer_athlete.mask
            
            # Ensure masks are boolean or binary
            m1_bool = (mask1 > 0)
            m2_bool = (mask2 > 0)
            
            # Calculate Intersection (pixels shared by both)
            intersection = np.logical_and(m1_bool, m2_bool).sum()
            
            # Calculate Area of the "Inner" (smaller/tested) person
            inner_area = m1_bool.sum()
            
            if inner_area == 0:
                return False
                
            overlap_ratio = intersection / inner_area
            
            # With masks, real people rarely overlap more than 10-20% even if standing close.
            # If overlap is > 40%, it's likely a double detection of the same person.
            return overlap_ratio > self.mask_overlap_threshold

        # --- STRATEGY 2: BBOX FALLBACK ---
        inner_bbox = self._get_bbox_coords(inner_athlete)
        outer_bbox = self._get_bbox_coords(outer_athlete)
        
        ix1 = max(inner_bbox[0], outer_bbox[0])
        iy1 = max(inner_bbox[1], outer_bbox[1])
        ix2 = min(inner_bbox[2], outer_bbox[2])
        iy2 = min(inner_bbox[3], outer_bbox[3])

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        inner_area = self.calculate_bbox_area(inner_bbox)
        
        if inner_area > 0:
            overlap_ratio = intersection_area / inner_area
            return overlap_ratio > self.containment_threshold
            
        return False

    def filter_front_row_athletes(
        self,
        athletes: List,
        frame_height: int,
        verbose: bool = False
    ) -> Tuple[List, List]:
        """
        Filtering using Overlap Containment + Y-Axis Clustering.
        """
        if not athletes:
            return [], []
                    
        # Sort by FEET POSITION (Y-max) - Lowest on screen (highest Y value) first
        sorted_indices = np.argsort([-self._get_bbox_coords(a)[3] for a in athletes])
        
        valid_indices = []
        rejected_by_overlap = []
        
        for i in sorted_indices:
            current_athlete = athletes[i]
            is_overlapped = False
            
            # Check against already accepted athletes (who are more in front)
            for valid_idx in valid_indices:
                front_athlete = athletes[valid_idx]
                
                # CHANGED: Pass full objects to check masks, not just boxes
                if self._is_contained(current_athlete, front_athlete):
                    is_overlapped = True
                    break
            
            if is_overlapped:
                rejected_by_overlap.append(athletes[i])
            else:
                valid_indices.append(i)

        # These are the candidates after removing heavy overlaps
        candidates = [athletes[i] for i in valid_indices]
        
        if verbose and rejected_by_overlap:
            print(f"  [Depth] Removed due to overlap: {[a.name for a in rejected_by_overlap]}")

        if len(candidates) < 2:
            return candidates, rejected_by_overlap

        # --- STEP 2: Clustering based on 'Feet' position (Y-Max) ---
        
        y_max_values = []
        for a in candidates:
            bbox = self._get_bbox_coords(a)
            y_max_values.append(bbox[3]) 
            
        y_max_values = np.array(y_max_values).reshape(-1, 1)

        # Check variance (if everyone is standing on the same line)
        y_range = np.max(y_max_values) - np.min(y_max_values)
        if y_range < (frame_height * 0.05):
            if verbose: print("  [Depth] Variance low, assuming single row.")
            return candidates, rejected_by_overlap

        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(y_max_values)
            
            center_0 = kmeans.cluster_centers_[0][0]
            center_1 = kmeans.cluster_centers_[1][0]
            
            # Larger Y value = Lower on screen = Front Row
            front_label = 0 if center_0 > center_1 else 1
            
            front_row = []
            back_row = list(rejected_by_overlap) 
            
            for i, athlete in enumerate(candidates):
                if labels[i] == front_label:
                    front_row.append(athlete)
                else:
                    back_row.append(athlete)
                    
            return front_row, back_row

        except Exception as e:
            print(f"  [Depth] Clustering failed ({e}), falling back to all-front.")
            return candidates, rejected_by_overlap