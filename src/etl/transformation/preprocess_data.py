import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import imagehash
import json


class ImageDeduplicator:
    """
    Deduplicates images using perceptual hashing.
    Supports multiple deduplication strategies:
    - Exact duplicates (identical hash)
    - Near duplicates (similar hash within threshold)
    """
    
    def __init__(self, source_dir, output_dir, hash_size=8, similarity_threshold=5, strategy='exact'):
        """
        Args:
            source_dir: Root directory containing athlete folders
            output_dir: Directory to save deduplicated images
            hash_size: Size of perceptual hash (8=64bit, 16=256bit). Larger = more precise
            similarity_threshold: Hamming distance threshold for near-duplicates (0-64 for hash_size=8)
            strategy: 'exact' (only identical), 'near' (similar within threshold), or 'aggressive' (very strict)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.strategy = strategy
        
        # Adjust threshold based on strategy
        if strategy == 'exact':
            self.similarity_threshold = 0
        elif strategy == 'aggressive':
            self.similarity_threshold = min(3, similarity_threshold)
        
        self.stats = {
            'total_processed': 0,
            'duplicates_found': 0,
            'unique_kept': 0,
            'per_athlete': defaultdict(lambda: {'total': 0, 'duplicates': 0, 'kept': 0})
        }
        
    def compute_hash(self, image_path):
        """Compute perceptual hash for an image."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB to handle different formats consistently
                img = img.convert('RGB')
                # Use average hash (fast and good for duplicates)
                # Alternative: phash (more robust) or dhash (good for transformations)
                return imagehash.average_hash(img, hash_size=self.hash_size)
        except Exception as e:
            print(f"Error hashing {image_path.name}: {e}")
            return None
    
    def find_duplicates(self, athlete_dir):
        """
        Find duplicate images within an athlete's folder.
        Returns dict mapping representative image to list of duplicates.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = []
        
        # Collect all images (including from subfolders)
        for root, dirs, files in os.walk(athlete_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)
        
        if not image_files:
            return {}
        
        print(f"\nProcessing {athlete_dir.name}: {len(image_files)} images")
        
        # Compute hashes
        hash_to_images = defaultdict(list)
        for img_path in tqdm(image_files, desc=f"Hashing {athlete_dir.name}"):
            img_hash = self.compute_hash(img_path)
            if img_hash is not None:
                hash_to_images[img_hash].append(img_path)
                self.stats['total_processed'] += 1
                self.stats['per_athlete'][athlete_dir.name]['total'] += 1
        
        # Find duplicates based on strategy
        duplicates = {}
        processed_hashes = set()
        
        all_hashes = list(hash_to_images.keys())
        
        for i, hash1 in enumerate(all_hashes):
            if hash1 in processed_hashes:
                continue
                
            similar_group = [hash1]
            
            # For near-duplicate detection, compare with other hashes
            if self.similarity_threshold > 0:
                for hash2 in all_hashes[i+1:]:
                    if hash2 in processed_hashes:
                        continue
                    
                    # Calculate Hamming distance
                    distance = hash1 - hash2
                    if distance <= self.similarity_threshold:
                        similar_group.append(hash2)
                        processed_hashes.add(hash2)
            
            # Collect all images in this group
            group_images = []
            for h in similar_group:
                group_images.extend(hash_to_images[h])
            
            if len(group_images) > 1:
                # Keep the first one as representative
                representative = group_images[0]
                duplicates[representative] = group_images[1:]
                
                self.stats['duplicates_found'] += len(group_images) - 1
                self.stats['per_athlete'][athlete_dir.name]['duplicates'] += len(group_images) - 1
        
        return duplicates
    
    def deduplicate_athlete(self, athlete_dir):
        """Deduplicate images for a single athlete and copy unique ones."""
        duplicates = self.find_duplicates(athlete_dir)
        
        # Create output directory for this athlete
        output_athlete_dir = self.output_dir / athlete_dir.name
        output_athlete_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all unique images
        all_images = set()
        for root, dirs, files in os.walk(athlete_dir):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
                    all_images.add(Path(root) / file)
        
        # Remove duplicates from the set
        duplicate_images = set()
        for dup_list in duplicates.values():
            duplicate_images.update(dup_list)
        
        unique_images = all_images - duplicate_images
        
        # Copy unique images (flatten structure)
        for img_path in unique_images:
            dest_path = output_athlete_dir / img_path.name
            
            # Handle name collisions
            counter = 1
            while dest_path.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                dest_path = output_athlete_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(img_path, dest_path)
        
        self.stats['unique_kept'] += len(unique_images)
        self.stats['per_athlete'][athlete_dir.name]['kept'] = len(unique_images)
        
        # Save duplicate report
        if duplicates:
            report_path = output_athlete_dir / "duplicates_report.json"
            report = {
                str(rep): [str(dup) for dup in dup_list] 
                for rep, dup_list in duplicates.items()
            }
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return len(unique_images)
    
    def run(self):
        """Process all athlete directories."""
        if not self.source_dir.exists():
            print(f"Error: Source directory not found: {self.source_dir}")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        athlete_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        
        if not athlete_dirs:
            print(f"No athlete directories found in {self.source_dir}")
            return
        
        print(f"Found {len(athlete_dirs)} athlete directories")
        print(f"Strategy: {self.strategy} (threshold: {self.similarity_threshold})")
        print("="*60)
        
        for athlete_dir in athlete_dirs:
            unique_count = self.deduplicate_athlete(athlete_dir)
            print(f"✓ {athlete_dir.name}: Kept {unique_count} unique images")
        
        self.print_summary()
    
    def print_summary(self):
        """Print deduplication statistics."""
        print("\n" + "="*60)
        print("DEDUPLICATION SUMMARY")
        print("="*60)
        print(f"Total images processed: {self.stats['total_processed']}")
        print(f"Duplicates found: {self.stats['duplicates_found']}")
        print(f"Unique images kept: {self.stats['unique_kept']}")
        print(f"Reduction: {self.stats['duplicates_found']/self.stats['total_processed']*100:.1f}%")
        print("\nPer-athlete breakdown:")
        for athlete, stats in self.stats['per_athlete'].items():
            reduction = stats['duplicates']/stats['total']*100 if stats['total'] > 0 else 0
            print(f"  {athlete}: {stats['kept']}/{stats['total']} kept ({reduction:.1f}% duplicates)")
        
        # Save summary to file
        summary_path = self.output_dir / "deduplication_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(dict(self.stats), f, indent=2, default=str)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate images across athlete directories before train/val/test split."
    )
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True, 
        help="Root directory containing athlete folders with images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save deduplicated images"
    )
    parser.add_argument(
        "--hash_size", 
        type=int, 
        default=8, 
        help="Perceptual hash size (8=fast, 16=precise). Default: 8"
    )
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=5, 
        help="Hamming distance threshold for near-duplicates (0-64 for hash_size=8). Default: 5"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=['exact', 'near', 'aggressive'], 
        default='near',
        help="Deduplication strategy: 'exact' (identical only), 'near' (similar), 'aggressive' (very strict)"
    )
    
    args = parser.parse_args()
    
    deduplicator = ImageDeduplicator(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        hash_size=args.hash_size,
        similarity_threshold=args.threshold,
        strategy=args.strategy
    )
    
    deduplicator.run()