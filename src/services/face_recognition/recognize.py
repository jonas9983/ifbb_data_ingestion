import os
import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def find_best_match(face_embedding, db, threshold=0.35):
    best_name = "Unknown"
    best_score = -1
    for name in db.files:
        db_emb = db[name]
        score = cosine_similarity(face_embedding, db_emb)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < threshold:
        best_name = "Unknown"
    return best_name, best_score

def recognize_faces_in_directory(data_dir, db_path="face_db.npz", threshold=0.35):
    # Load face database
    db = np.load(db_path)

    # Initialize InsightFace
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Create processed folder
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Iterate over all images in the folder
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)

        if not os.path.isfile(img_path):
            continue
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")) is False:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"No faces detected in {img_name}")
        else:
            for face in faces:
                name, score = find_best_match(face.embedding, db, threshold)
                box = face.bbox.astype(int)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        output_path = os.path.join(processed_dir, img_name)
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch face recognition on a folder using InsightFace embeddings.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--db", type=str, default="face_db.npz", help="Face database npz path")
    parser.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity threshold")

    args = parser.parse_args()

    recognize_faces_in_directory(args.data_dir, db_path=args.db, threshold=args.threshold)
