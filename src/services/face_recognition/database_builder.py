import os
import numpy as np
import cv2
import argparse
from insightface.app import FaceAnalysis


def build_face_database(data_dir, save_path):
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    database = {}

    for person in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, person)
        if not os.path.isdir(folder_path):
            continue

        embeddings = []

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            face = max(faces, key=lambda f: f.det_score)
            embeddings.append(face.embedding)

        if len(embeddings) == 0:
            continue

        avg_emb = np.mean(np.array(embeddings), axis=0)
        database[person] = avg_emb

    np.savez(save_path, **database)
    return database


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a face embedding database.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing subfolders of person images.")
    parser.add_argument("--save_path", type=str, default="face_db.npz",
                        help="Output .npz file path.")

    args = parser.parse_args()

    build_face_database(args.data_dir, args.save_path)
    print(f"Database saved to {args.save_path}")
