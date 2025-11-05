import os
import random
import shutil
from pathlib import Path
import argparse


class DatasetSplitter:
    def __init__(self, source_dir, dest_dir, max_images_total=200, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.max_images_total = max_images_total
        self.split_sizes = {
            "train": int(max_images_total * train_ratio),
            "val": int(max_images_total * val_ratio),
            "test": int(max_images_total * test_ratio)
        }

        for split in self.split_sizes.keys():
            (self.dest_dir / split).mkdir(parents=True, exist_ok=True)

    def run(self):
        athletes = [d for d in self.source_dir.iterdir() if d.is_dir()]
        for athlete_dir in athletes:
            self._process_athlete(athlete_dir)
        print("Dataset split completed successfully.")

    def _process_athlete(self, athlete_dir):
        # Collect sources
        frame_folders = [f for f in athlete_dir.iterdir() if f.is_dir()]
        standalone_images = list(athlete_dir.glob("*.jpg"))
        random.shuffle(frame_folders)
        random.shuffle(standalone_images)

        # Split allocation counters
        split_counts = {"train": 0, "val": 0, "test": 0}
        max_per_split = self.split_sizes
        split_data = {"train": [], "val": [], "test": []}

        # First handle standalone images (random shuffle)
        for img_path in standalone_images:
            if sum(split_counts.values()) >= self.max_images_total:
                break
            split_name = self._choose_split(split_counts, max_per_split)
            if split_name:
                split_data[split_name].append(img_path)
                split_counts[split_name] += 1

        # Then handle frame folders (keep folder consistency)
        for folder in frame_folders:
            images = list(folder.glob("*.jpg"))
            random.shuffle(images)

            # If folder has no images, skip
            if not images:
                continue

            # Choose which split this folder should go to
            split_name = self._choose_split(split_counts, max_per_split)
            if not split_name:
                break

            for img_path in images:
                if split_counts[split_name] >= max_per_split[split_name] or \
                   sum(split_counts.values()) >= self.max_images_total:
                    break
                split_data[split_name].append(img_path)
                split_counts[split_name] += 1

        # Copy data
        for split_name, img_list in split_data.items():
            dest_athlete_dir = self.dest_dir / split_name / athlete_dir.name
            dest_athlete_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_list:
                shutil.copy(img_path, dest_athlete_dir / img_path.name)

            print(f"{athlete_dir.name}: {split_name} -> {len(img_list)} images")

    def _choose_split(self, split_counts, max_per_split):
        """Choose which split to assign next item to based on available quota."""
        available = [s for s, count in split_counts.items() if count < max_per_split[s]]
        if not available:
            return None
        # Prefer filling train first, then val, then test
        for s in ["train", "val", "test"]:
            if s in available:
                return s
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset by athlete into train/val/test sets (fixed 200 total).")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the root dataset directory.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path to the output split dataset directory.")
    parser.add_argument("--max_images_total", type=int, default=200, help="Total number of images per athlete.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (default 0.8).")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (default 0.1).")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio (default 0.1).")

    args = parser.parse_args()

    splitter = DatasetSplitter(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        max_images_total=args.max_images_total,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    splitter.run()
