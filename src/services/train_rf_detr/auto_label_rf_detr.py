import argparse
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rfdetr import RFDETRMedium
import torch
import numpy as np

ALL_USER_FOLDERS = [
    "ahmd_ashkanani",
    "angelcalderonfrias",
    "fabriciomoreirapro",
    "hideyamagishi",
    "keone_prodigy",
    "kerrith_bajjo",
    "melnikovifbb",
    "nassersayed_ifbbpro",
    "olehkryvyi",
    "shaunclarida",
    "theradoslavangelov",
    "venom_from_ukraine",
    "vitorportopro"
]


class RF_Detr_AutoLabeler:
    BASE_COCO_PERSON_CLASS_ID = 1

    # Added 'target_user_name' argument for flexibility
    def __init__(self, model, input_dir, split_name, target_user_name, confidence_threshold=0.5, image_url_prefix=""):
        self.input_path = Path(input_dir).resolve()
        self.confidence_threshold = confidence_threshold
        self.image_url_prefix = image_url_prefix
        self.model = model
        self.split_name = split_name  # Store the split name (train/val/test)

        self.target_user = target_user_name  # <<< Use explicit name
        self.categories_map = {name: i + 1 for i, name in enumerate(ALL_USER_FOLDERS)}
        
        # Check if the target user is a known athlete
        if self.target_user in self.categories_map:
            self.TARGET_CATEGORY_ID = self.categories_map[self.target_user]
            self.is_athlete_folder = True # Flag for path formatting
        else:
            # This is the "flat" case (e.g., target_user is 'test')
            self.TARGET_CATEGORY_ID = None # Not a specific athlete
            self.is_athlete_folder = False # Flag for path formatting
            print(f"Info: Processing flat directory. Labeling detected people as '{self.target_user}'.")

    def _get_image_files(self):
        if not self.input_path.is_dir():
            print(f"Error: Input directory not found at {self.input_path}")
            return []

        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in self.input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {self.input_path}.")
        return image_files

    def _format_task(self, img_file, detections):
        try:
            with Image.open(img_file) as img:
                original_width, original_height = img.size
        except Exception as e:
            print(f"Error reading image {img_file.name}: {e}. Skipping.")
            return None

        if self.is_athlete_folder:
            # Original case: split/athlete/image.jpg
            relative_storage_path = f"{self.split_name}/{self.target_user}/{img_file.name}"
        else:
            # Flat case: split/image.jpg
            relative_storage_path = f"{self.split_name}/{img_file.name}"

        local_file_url_part = f"/data/local-files/?d={relative_storage_path}"

        if self.image_url_prefix:
            base_url = self.image_url_prefix.rstrip('/')
            image_url = f"{base_url}{local_file_url_part}"
        else:
            image_url = local_file_url_part

        task = {
            "data": {"image": image_url},
            "predictions": []
        }

        if detections and len(detections.xyxy) > 0:
            prediction_results = []

            for i in range(len(detections.xyxy)):
                bbox_xyxy = detections.xyxy[i]
                confidence = detections.confidence[i]

                if isinstance(bbox_xyxy, torch.Tensor):
                    bbox_xyxy = bbox_xyxy.cpu().numpy()
                elif not isinstance(bbox_xyxy, np.ndarray):
                    bbox_xyxy = np.array(bbox_xyxy)

                if isinstance(confidence, torch.Tensor):
                    confidence = confidence.item()
                elif isinstance(confidence, np.ndarray):
                    confidence = float(confidence)
                else:
                    confidence = float(confidence)

                x_min, y_min, x_max, y_max = bbox_xyxy

                x_percent = (float(x_min) / original_width) * 100
                y_percent = (float(y_min) / original_height) * 100
                width_percent = ((float(x_max) - float(x_min)) / original_width) * 100
                height_percent = ((float(y_max) - float(y_min)) / original_height) * 100

                result_item = {
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": width_percent,
                        "height": height_percent,
                        "rotation": 0,
                        # The label is now flexible: 'ahmd_ashkanani' or 'test'
                        "rectanglelabels": [self.target_user] 
                    },
                    "id": f"bbox_{i}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels"
                }

                prediction_results.append(result_item)

            if prediction_results:
                task["predictions"].append({
                    "result": prediction_results,
                    "score": float(np.mean([
                        float(detections.confidence[i])
                        if isinstance(detections.confidence[i], (np.ndarray, torch.Tensor))
                        else detections.confidence[i]
                        for i in range(len(detections.confidence))
                    ]))
                })

        return task

    def run(self):
        image_files = self._get_image_files()
        if not image_files:
            return []

        all_tasks = []

        # tqdm description now correctly shows 'ahmd_ashkanani' or 'test'
        for img_file in tqdm(image_files, desc=f"Inference: {self.target_user}"):
            try:
                image = Image.open(img_file).convert("RGB")
                detections = self.model.predict(image, threshold=self.confidence_threshold)
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}. Skipping.")
                continue

            class_id = detections.class_id
            if isinstance(class_id, torch.Tensor):
                person_mask = class_id == self.BASE_COCO_PERSON_CLASS_ID
            elif isinstance(class_id, np.ndarray):
                person_mask = class_id == self.BASE_COCO_PERSON_CLASS_ID
            else:
                person_mask = [cid == self.BASE_COCO_PERSON_CLASS_ID for cid in class_id]

            person_detections = detections[person_mask]

            task = self._format_task(img_file, person_detections)
            if task:
                all_tasks.append(task)

        return all_tasks

def process_entire_dataset(base_dir, output_dir, confidence_threshold=0.5, image_url_prefix=""):
    base_dir = Path(base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    splits = ["train", "val", "test"]

    print(f"Running auto-labeling for dataset in {base_dir}")

    # Ensure model is initialized only once
    model = RFDETRMedium()
    model.optimize_for_inference()

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_path = base_dir / split
        if not split_path.exists():
            print(f"Split not found: {split_path}")
            continue

        combined_tasks = []
        # List will store (input_dir_path, target_user_name, descriptive_name)
        directories_to_process = []
        
        # Check if the split directory contains any subdirectories
        subdirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # Case 1: Subfolders (e.g., /train/<athlete_name>) exist.
            # Target user is the subdir name.
            for athlete_dir in subdirs:
                if athlete_dir.name in ALL_USER_FOLDERS:
                    directories_to_process.append((athlete_dir, athlete_dir.name, athlete_dir.name))
        else:
            # Case 2: No subfolders found (images are directly in /train/).
            # Target user is the split name (e.g., 'train').
            target_name = split_path.name
            directories_to_process.append((split_path, target_name, f"root ({target_name})"))

        if not directories_to_process:
            print(f"Skipping {split}: No relevant images or subfolders found.")
            continue

        # Unpack the 3-item tuple
        for input_dir, target_user_name, descriptive_name in directories_to_process:
            print(f"Processing {split}/{descriptive_name}")
            
            labeler = RF_Detr_AutoLabeler(
                model=model,
                input_dir=input_dir,
                split_name=split,
                target_user_name=target_user_name,
                confidence_threshold=confidence_threshold,
                image_url_prefix=image_url_prefix
            )
            tasks = labeler.run()
            combined_tasks.extend(tasks)
        
        output_path = output_dir / f"{split}.json"
        with open(output_path, 'w') as f:
            json.dump(combined_tasks, f, indent=2)

        print(f"Saved {len(combined_tasks)} tasks to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Auto-label train/val/test images using RFDETR and save combined results per split.'
    )
    parser.add_argument('--input', required=True, help='Root directory containing train/val/test folders.')
    parser.add_argument('--output', default='data/annotations', help='Directory to save combined JSONs.')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default=0.5)')
    parser.add_argument('--url-prefix', type=str, default="", help='Optional URL prefix for Label Studio.')

    args = parser.parse_args()

    process_entire_dataset(
        base_dir=args.input,
        output_dir=args.output,
        confidence_threshold=args.conf,
        image_url_prefix=args.url_prefix
    )