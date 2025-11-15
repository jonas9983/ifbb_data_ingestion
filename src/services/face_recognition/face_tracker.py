import os
import cv2
import numpy as np
import argparse
from itertools import combinations
from sort.sort import Sort
from src.services.face_recognition.face_recognizer import FaceRecognizer


def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two boxes."""
    # [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def find_face_for_track(faces, track_box):
    """
    Finds the original InsightFace 'face' object that best matches
    a SORT tracked box using IoU.
    """
    best_iou = 0
    best_face = None
    
    for face in faces:
        iou = calculate_iou(face.bbox, track_box)
        if iou > best_iou:
            best_iou = iou
            best_face = face
            
    # We require a minimum IoU to consider it a match
    if best_iou > 0.5:
        return best_face
    return None


def main(args):
    # 1. Initialize the FaceRecognizer
    # We use this for both detection (app.get) and recognition (match)
    print("Loading FaceRecognizer...")
    recognizer = FaceRecognizer(args.db, threshold=args.threshold)
    print("Recognizer loaded.")

    # 2. Initialize the SORT tracker
    # You can tune these parameters
    # max_age: Frames to keep a "lost" track
    # min_hits: Frames to wait before "confirming" a new track
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    # --- State Variables ---
    # This dictionary will map a track_id (from SORT) to a name (from FaceRecognizer)
    track_id_to_name = {}
    
    # This dictionary stores the relative order of pairs, e.g., {(1, 2): '1_left_of_2'}
    last_known_relative_order = {}
    
    # --- Setup Output Directory ---
    processed_dir = os.path.join(args.data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(args.data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    print(f"Starting processing on {len(images)} images...")
    
    for img_name in images:
        img_path = os.path.join(args.data_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. DETECTION: Get all faces from InsightFace
        # 'faces' is a list of [Face] objects. Each has .bbox, .det_score, .embedding
        faces = recognizer.app.get(img)
        
        # 2. TRACKING (Part A): Format detections for SORT
        # SORT needs a numpy array of [x1, y1, x2, y2, score]
        detections_for_sort = []
        for face in faces:
            detections_for_sort.append(list(face.bbox) + [face.det_score])
        
        if len(detections_for_sort) == 0:
            # If no faces, just update tracker with empty array and save image
            tracker.update(np.empty((0, 5)))
            cv2.imwrite(os.path.join(processed_dir, img_name), img)
            continue
            
        detections_for_sort = np.array(detections_for_sort)

        # 3. TRACKING (Part B): Update SORT
        # 'tracked_objects' is a N x 5 array: [x1, y1, x2, y2, track_id]
        tracked_objects = tracker.update(detections_for_sort)

        # --- This holds the tracks visible in the *current* frame ---
        current_frame_tracks = {}

        # 4. RECOGNITION: Link Track IDs to Names
        for track in tracked_objects:
            box = track[:4].astype(int)
            track_id = int(track[4])
            
            if track_id not in track_id_to_name:
                matched_face = find_face_for_track(faces, box)
                if matched_face:
                    name, score = recognizer.match(matched_face.embedding)
                    track_id_to_name[track_id] = name
                    print(f"[New Track]: ID {track_id} has been recognized as {name}")
            else:
                # Re-recognize if currently unknown
                if track_id_to_name[track_id] == "Unknown":
                    matched_face = find_face_for_track(faces, box)
                    if matched_face:
                        name, score = recognizer.match(matched_face.embedding)
                        # Only update if we get a confident match (not Unknown)
                        if name != "Unknown":
                            track_id_to_name[track_id] = name
                            print(f"[Updated Track]: ID {track_id} updated from Unknown to {name}")
            
            # Get the name (either new or from memory)
            name = track_id_to_name.get(track_id, "Tracking...")
            
            # Store data for swap detection and drawing
            current_frame_tracks[track_id] = {
                'name': name,
                'center_x': (box[0] + box[2]) / 2,
                'box': box
            }

        # 5. SWAP DETECTION
        visible_track_ids = sorted(list(current_frame_tracks.keys()))
        
        if len(visible_track_ids) >= 2:
            # Check all unique pairs of visible people
            for id_a, id_b in combinations(visible_track_ids, 2):
                pair_key = (id_a, id_b) # Always sorted (e.g., (1, 3))
                
                pos_a = current_frame_tracks[id_a]['center_x']
                pos_b = current_frame_tracks[id_b]['center_x']
                
                # Determine current order
                current_order = f"{id_a}_left_of_{id_b}" if pos_a < pos_b else f"{id_b}_left_of_{id_a}"
                
                # Compare to last known order
                last_order = last_known_relative_order.get(pair_key)
                
                if last_order and current_order != last_order:
                    name_a = current_frame_tracks[id_a]['name']
                    name_b = current_frame_tracks[id_b]['name']
                    print("==================================================")
                    print(f" POSITION SWAP DETECTED IN: {img_name}")
                    print(f"   Between: {name_a} (ID {id_a}) and {name_b} (ID {id_b})")
                    print(f"   Previous: {last_order}")
                    print(f"   New:      {current_order}")
                    print("==================================================")

                # Update the last known order for this pair
                last_known_relative_order[pair_key] = current_order
                
        # 6. DRAWING
        for track_id, data in current_frame_tracks.items():
            box = data['box']
            label = f"ID {track_id} ({data['name']})"
            
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(processed_dir, img_name), img)

    print("---")
    print("Tracking and swap detection complete.")
    print(f"Total unique people tracked: {len(track_id_to_name)}")
    print("Track ID to Name mapping:")
    for tid, name in track_id_to_name.items():
        print(f"  ID {tid}: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track faces using SORT and detect position swaps.")
    
    # Arguments from your original script
    parser.add_argument("--data_dir", required=True, help="Directory containing image frames.")
    parser.add_argument("--db", required=True, help="Path to the saved face database (.npz file).")
    parser.add_argument("--threshold", type=float, default=0.35, help="Recognition cosine similarity threshold.")
    
    args = parser.parse_args()
    main(args)