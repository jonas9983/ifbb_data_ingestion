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
from src.services.faces.face_recognizer import FaceDatabaseBuilder

class SingleFaceFilter:
    """Filters images to only those with exactly one face."""
    
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
        # This discards the directories that are inside which are frames from the video posts.
        # The data collection now is different so it makes more sense to do it like this        
        single_face_count = 0
        
        for img_name in images:
            img_path = os.path.join(source_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            # Get faces from the img            
            faces = self.app.get(img)
            
            if len(faces) == 1:
                target_path = os.path.join(athlete_validation_dir, img_name)
                cv2.imwrite(target_path, img)
                single_face_count += 1
                
                if single_face_count >= self.max_images:
                    print(f"Max images ({self.max_images}) Reached")
                    return

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
        
        # Build database with image references
        database = self._build_database_with_references()
        
        print(f"✓ Database saved: {self.db_save_path}")
        print(f"  Athletes in database: {len(database)}")
        return True
    
    def _build_database_with_references(self):
        """Build face database and track which images were used."""
        builder = FaceDatabaseBuilder()
        database = builder.build(self.database_dir, self.db_save_path, save_references=True)
        return database
    
    def run_full_workflow(self, skip_scrape=False, skip_filter=False, 
                          skip_validation=False, skip_build=False):
        """Run the complete workflow with optional step skipping."""
        
        if not skip_scrape:
            if not self.step1_scrape_instagram():
                print("Scraping failed or skipped")
                return
        
        if not skip_filter:
            self.step2_filter_single_faces()
        
        if not skip_validation:
            self.step3_manual_validation()
        
        if not skip_build:
            self.step4_build_database()
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)


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