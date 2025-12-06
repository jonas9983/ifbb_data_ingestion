import os
import yaml
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from apify_client import ApifyClient

# Load environment variables
load_dotenv()
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

class BillingGuard:
    """Protects your wallet by checking usage before running."""
    
    HARD_LIMIT_USD = 4.80  # Stop if total spend is above this (Max budget is $5.00)

    @staticmethod
    def can_run(api_token: str) -> bool:
        url = "https://api.apify.com/v2/users/me/usage/monthly"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json().get('data', {})
            
            total_spend = data.get('totalCostUsd', 0)
            print(f"Current Monthly Spend: ${total_spend:.2f} / $5.00")
            
            if total_spend >= BillingGuard.HARD_LIMIT_USD:
                print(f"STOPPING: Budget limit reached (${BillingGuard.HARD_LIMIT_USD}).")
                return False
            
            remaining = 5.00 - total_spend
            print(f"Budget OK. Remaining: ${remaining:.2f}")
            return True
            
        except Exception as e:
            print(f"Could not verify billing: {e}. Proceeding with caution.")
            return True 

class InstagramScraper:
    def __init__(self, config_path: str = "config.yaml"):
        self.client = ApifyClient(APIFY_API_TOKEN)
        self.config = self._load_config(config_path)
        self.actor_id = "apify/instagram-post-scraper"

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def run(self):
        # 1. Global Budget Check
        if not BillingGuard.can_run(APIFY_API_TOKEN):
            return

        usernames = self.config['athletes']
        limit = self.config['settings']['max_posts_per_user']
        download_dir = self.config['settings']['download_folder']

        print(f"Target List: {usernames}")

        # 2. Iterate and Scrape One by One
        for athlete in usernames:
            print(f"--- Processing: {athlete} ---")
            
            run_input = {
                "username": [athlete], # Pass only the current athlete
                "resultsLimit": limit,
                "searchType": "posts",
                "searchLimit": 1, 
            }

            print(f"Requesting data for {athlete}...")
            
            # Run the actor specifically for this one athlete
            run = self.client.actor(self.actor_id).call(run_input=run_input)
            
            if not run:
                print(f"Failed to get data for {athlete}")
                continue

            dataset_id = run['defaultDatasetId']
            
            # 3. Download results immediately to the specific folder
            self._download_images(dataset_id, download_dir, athlete)

    def _download_images(self, dataset_id: str, base_dir: str, folder_name: str):
        """Forces all images from this dataset into base_dir/folder_name."""
        
        # Create the specific folder for this athlete (from YAML name)
        target_dir = os.path.join(base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Downloading images to: {target_dir}")
        
        dataset_items = self.client.dataset(dataset_id).iterate_items()
        
        count = 0
        for item in dataset_items:
            url = item.get('displayUrl')
            
            if not url:
                continue
            
            # Use post ID for filename
            filename = f"{item.get('id', 'post')}.jpg"
            file_path = os.path.join(target_dir, filename)
            
            if not os.path.exists(file_path):
                try:
                    img_data = requests.get(url).content
                    with open(file_path, 'wb') as f:
                        f.write(img_data)
                    count += 1
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

        print(f"Saved {count} images for {folder_name}")

if __name__ == "__main__":
    scraper = InstagramScraper()
    scraper.run()