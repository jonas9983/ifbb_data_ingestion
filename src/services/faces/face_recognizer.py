import os
import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
from numpy import dot
from numpy.linalg import norm


class FaceDatabaseBuilder:
    def __init__(self, providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def build(self, data_dir, save_path, save_references=False):
        """
        Build face database from directory of athlete folders.
        
        Args:
            data_dir: Directory containing subdirectories for each person
            save_path: Path to save the .npz database file
            save_references: If True, save image filenames used for each athlete
        
        Returns:
            database: Dict mapping person names to embeddings
        """
        database = {}
        image_references = {} if save_references else None

        for person in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, person)
            if not os.path.isdir(folder_path):
                continue

            embeddings = []
            used_images = [] if save_references else None

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = self.app.get(img)
                if len(faces) == 0:
                    continue

                face = max(faces, key=lambda f: f.det_score)
                embeddings.append(face.embedding)
                
                if save_references:
                    used_images.append(img_name)

            if len(embeddings) == 0:
                continue

            avg_emb = np.mean(np.array(embeddings), axis=0)
            database[person] = avg_emb
            
            if save_references:
                image_references[person] = used_images

        # Save to .npz file
        if save_references:
            save_dict = {}
            for name, emb in database.items():
                save_dict[name] = emb
                save_dict[f"{name}_images"] = np.array(image_references[name], dtype=object)
            np.savez(save_path, **save_dict)
        else:
            np.savez(save_path, **database)
        
        return database


class FaceRecognizer:
    def __init__(self, db_path, threshold=0.35, providers=['CPUExecutionProvider']):
        self.db = np.load(db_path, allow_pickle=True)
        self.threshold = threshold

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def match(self, embedding):
        best_name = "Unknown"
        best_score = -1

        for name in self.db.files:
            # Skip image reference entries
            if name.endswith('_images'):
                continue
                
            score = self.cosine_similarity(embedding, self.db[name])
            if score > best_score:
                best_name = name
                best_score = score

        if best_score < self.threshold:
            best_name = "Unknown"

        return best_name, best_score
    
    def get_image_references(self, athlete_name):
        """Get the list of images used to build an athlete's embedding."""
        key = f"{athlete_name}_images"
        if key in self.db.files:
            return self.db[key].tolist()
        return None

    def process_directory(self, data_dir):
        processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        for img_name in images:
            img_path = os.path.join(data_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = self.app.get(img)

            for face in faces:
                name, score = self.match(face.embedding)
                box = face.bbox.astype(int)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imwrite(os.path.join(processed_dir, img_name), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--db", required=False)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--build_db", action="store_true")
    parser.add_argument("--save_db_path", default="face_db.npz")
    parser.add_argument("--save_references", action="store_true", 
                        help="Save image filenames used for each athlete")
    args = parser.parse_args()

    if args.build_db:
        builder = FaceDatabaseBuilder()
        builder.build(args.data_dir, args.save_db_path, save_references=args.save_references)
    else:
        recognizer = FaceRecognizer(args.db, threshold=args.threshold)
        recognizer.process_directory(args.data_dir)