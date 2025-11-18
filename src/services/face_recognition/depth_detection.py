import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans

class DepthAnalyzer:
    """
    Determines if athletes are in the front row or back row using
    Y-Max (Feet) clustering and Containment checks.
    """
    
    def __init__(self):
        # Minimum intersection-over-area to consider a box "contained" in another
        self.containment_threshold = 0.6 

    def _get_bbox_coords(self, athlete):
        """Helper to safely get bbox [x1, y1, x2, y2]."""
        # Prefer person body box, fallback to face
        bbox = athlete.person_bbox if athlete.person_bbox else athlete.face_bbox
        return bbox

    def calculate_bbox_area(self, bbox: List[int]) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _is_contained(self, inner_bbox, outer_bbox) -> bool:
        """
        Check if inner_bbox is significantly overlapping/inside outer_bbox.
        """
        ix1 = max(inner_bbox[0], outer_bbox[0])
        iy1 = max(inner_bbox[1], outer_bbox[1])
        ix2 = min(inner_bbox[2], outer_bbox[2])
        iy2 = min(inner_bbox[3], outer_bbox[3])

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        inner_area = self.calculate_bbox_area(inner_bbox)
        
        # If > 60% of the smaller box is inside the larger box, it's a background ghost
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
        Robust filtering using Overlap Containment + Y-Axis Clustering.
        """
        if not athletes:
            return [], []
                    
        # Sort by area (largest first) to identify the "blockers"
        sorted_indices = np.argsort([-self.calculate_bbox_area(self._get_bbox_coords(a)) for a in athletes])
        
        valid_indices = []
        rejected_by_overlap = []
        
        for i in sorted_indices:
            current_bbox = self._get_bbox_coords(athletes[i])
            is_overlapped = False
            
            # Check against already accepted larger athletes
            for valid_idx in valid_indices:
                larger_bbox = self._get_bbox_coords(athletes[valid_idx])
                if self._is_contained(current_bbox, larger_bbox):
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
        # Front row feet are lower (higher pixel value) than back row feet.
        
        y_max_values = []
        for a in candidates:
            bbox = self._get_bbox_coords(a)
            y_max_values.append(bbox[3]) # The bottom coordinate
            
        y_max_values = np.array(y_max_values).reshape(-1, 1)

        # If the spread of feet positions is small (e.g. < 5% of frame height),
        # assume everyone is in the same row (Front Row).
        y_range = np.max(y_max_values) - np.min(y_max_values)
        if y_range < (frame_height * 0.05):
            if verbose: print("  [Depth] Variance low, assuming single row.")
            return candidates, rejected_by_overlap

        # Use K-Means to find 2 clusters (Front vs Back)
        # We expect at least 2 distinct Y-levels if there is a back row.
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(y_max_values)
            
            # Determine which cluster is the "Front"
            # The cluster with the HIGHER average y_max value is the front (bottom of screen)
            center_0 = kmeans.cluster_centers_[0][0]
            center_1 = kmeans.cluster_centers_[1][0]
            
            front_label = 0 if center_0 > center_1 else 1
            
            front_row = []
            back_row = list(rejected_by_overlap) # Include the overlapped ones in back row
            
            for i, athlete in enumerate(candidates):
                if labels[i] == front_label:
                    front_row.append(athlete)
                else:
                    back_row.append(athlete)
                    
            return front_row, back_row

        except Exception as e:
            print(f"  [Depth] Clustering failed ({e}), falling back to all-front.")
            return candidates, rejected_by_overlap
        