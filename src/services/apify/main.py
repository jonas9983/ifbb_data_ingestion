import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*rcond.*")

import os
import cv2
import yaml
import shutil
import argparse
from typing import Dict, Any
from insightface.app import FaceAnalysis

from src.etl.extraction.apify_extraction import InstagramScraper, BillingGuard, APIFY_API_TOKEN

class SingleFaceFilter:
    """Filters images to find the highest quality single-face shots."""
    
    def __init__(self, min_images: int, max_images: int, providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.min_images = min_images
        self.max_images = max_images
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def filter_directory(self, source_dir: str, target_dir: str, athlete_name: str):
        athlete_validation_dir = os.path.join(target_dir, athlete_name)
        os.makedirs(athlete_validation_dir, exist_ok=True)
        
        if not os.path.exists(source_dir):
            print(f"⚠ Source not found: {source_dir}")
            return 0
        
        images = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 
        
        valid_faces = []
        
        print(f"Scanning {len(images)} raw images for {athlete_name}...")
        
        for img_name in images:
            img_path = os.path.join(source_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            faces = self.app.get(img)
            
            # Only process images where there is exactly ONE face detected
            if len(faces) == 1:
                face = faces[0]
                
                bbox = face.bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_area = face_width * face_height
                
                # Baseline Check: Must be decent size and good confidence
                if face_width > 80 and face_height > 80 and face.det_score > 0.6:
                    
                    # Create a quality score: Bigger face + Higher confidence = Better
                    quality_score = face_area * face.det_score
                    
                    valid_faces.append({
                        'img_name': img_name,
                        'img_data': img,
                        'score': quality_score
                    })
        
        # Sort all found faces by their quality score (Highest to lowest)
        valid_faces.sort(key=lambda x: x['score'], reverse=True)
        
        # Slice the list to keep only the Top K images
        best_faces = valid_faces[:self.max_images]
        
        print(f" -> Found {len(valid_faces)} valid faces. Keeping top {len(best_faces)}.")
        
        # Save only the absolute best ones to the validation folder
        for face_data in best_faces:
            target_path = os.path.join(athlete_validation_dir, face_data['img_name'])
            cv2.imwrite(target_path, face_data['img_data'])
            
        return len(best_faces)

class DatabaseWorkflow:
    """Orchestrates the complete database building workflow."""
    
    def __init__(self, config_path: str = "./configs/apify.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.min_images = self.config['settings'].get('min_images_per_athlete', 3)
        self.max_images = self.config['settings'].get("max_images_per_athlete", 50)
        self.download_folder = self.config['settings']['download_folder']
        self.validation_dir = self.config['settings'].get('validation_folder', 'validation')
        self.database_dir = self.config['settings'].get('database_folder', 'database')
        self.npz_basename = self.config['settings'].get('npz_basename', 'face_db.npz')
        self.db_save_path = f"{self.database_dir}/{self.npz_basename}"
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    
    def step1_scrape_instagram(self):
        print("\n" + "="*60)
        print("STEP 1: SCRAPING INSTAGRAM")
        print("="*60)
        
        if not BillingGuard.can_run(APIFY_API_TOKEN):
            return False
        
        scraper = InstagramScraper(self.config_path)
        scraper.run()
        print("✓ Instagram scraping complete\n")
        return True
    
    def step2_filter_single_faces(self):
        print("\n" + "="*60)
        print("STEP 2: FILTERING SINGLE-FACE IMAGES")
        print("="*60)
        
        filter_tool = SingleFaceFilter(min_images = self.min_images, max_images = self.max_images)
        athletes = self.config['athletes']
        
        for athlete in athletes:
            print(f"Processing {athlete}")
            source_dir = os.path.join(self.download_folder, athlete)
            filter_tool.filter_directory(source_dir = source_dir, target_dir = self.validation_dir, athlete_name = athlete)
        
        print(f"\n✓ Filtering complete. Review: {self.validation_dir}/\n")
    
    def step3_manual_validation(self):
        print("\n" + "="*60)
        print("STEP 3: MANUAL VALIDATION")
        print("="*60)
        print(f"\n Validate images in: {self.validation_dir}/")
        input("Press Enter when ready to build database...")
        shutil.copytree(src = self.validation_dir, dst = self.database_dir)
    
    def step4_build_database(self):
        print("\n" + "="*60)
        print("STEP 4: BUILDING FACE DATABASE")
        print("="*60)
        
        if not os.path.exists(self.database_dir):
            print(f"⚠ Database directory not found: {self.database_dir}")
            os.makedirs(self.database_dir, exist_ok=True)
            return False
        
        subdirs = [d for d in os.listdir(self.database_dir) 
                   if os.path.isdir(os.path.join(self.database_dir, d))]
        
        if len(subdirs) == 0:
            print(f"⚠ No athlete folders found in {self.database_dir}")
            return False
        
        # Check minimum images for each athlete in database folder
        athletes_below_min = {}
        for athlete in subdirs:
            athlete_path = os.path.join(self.database_dir, athlete)
            img_count = len([f for f in os.listdir(athlete_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if img_count < self.min_images:
                athletes_below_min[athlete] = img_count
        
        if athletes_below_min:
            print(f"⚠ Warning: Some athletes have fewer than {self.min_images} validated images:")
            for name, count in athletes_below_min.items():
                print(f"    {name}: {count} images")
            
            response = input(f"\nContinue building database anyway? (y/n): ")
            if response.lower() != 'y':
                print("Database building cancelled.")
                return False
        
        print(f"Building database from {len(subdirs)} athletes...")
        
        
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database Building Workflow")
    parser.add_argument("--config", default="./configs/apify.yaml", help="Path to config file")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip Instagram scraping")
    parser.add_argument("--skip-filter", action="store_true", help="Skip face filtering")
    parser.add_argument("--skip-validation", action="store_true", help="Skip manual validation prompt")
    parser.add_argument("--skip-build", action="store_true", help="Skip database building")
    
    args = parser.parse_args()
    
    workflow = DatabaseWorkflow(args.config)
    workflow.run_full_workflow(
        skip_scrape=args.skip_scrape,
        skip_filter=args.skip_filter,
        skip_validation=args.skip_validation,
        skip_build=args.skip_build
    )