import json
import os
import requests
from pathlib import Path
from urllib.parse import urlparse
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

class InstagramDownloader:
    """
    Downloads media from a JSON taken from Apify and organizes it by specified target users.
    """
    
    def __init__(self, json_file, output_dir="instagram_downloads", target_users=None, workers=4):
        """
        Initializes the InstagramDownloader with configuration.

        Args:
            json_file (str): Path to JSON file from Apify.
            output_dir (str): Base directory for downloads.
            target_users (list/set): Usernames to track.
            workers (int): Number of parallel download threads.
        """
        self.json_file = json_file
        self.output_dir = output_dir
        # Ensure target_users is a set for O(1) lookups
        self.target_users = set(target_users) if target_users else set()
        self.workers = workers
        self.stats = {
            'total_posts': 0,
            'relevant_posts': 0,
            'images_downloaded': 0,
            'videos_downloaded': 0,
            'failed_downloads': 0,
            'skipped': 0
        }
        self.user_stats = {}
        self.download_tasks = []

    def _sanitize_filename(self, filename):
        """Helper to remove invalid characters from filename"""
        # Kept as a static utility method if preferred, but useful as a private method too
        return "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()

    def _download_file(self, url, filepath):
        """Helper to download a file from URL to filepath"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            return False

    def _get_target_users_in_post(self, post):
        """
        Get list of target users that appear in this post (as owner or tagged/mentioned).
        """
        found_users = []
        target_users = self.target_users
        
        # Check if owner is a target user
        owner = post.get('ownerUsername')
        if owner in target_users:
            found_users.append(owner)
        
        # Check tagged users
        for tagged_user in post.get('taggedUsers', []):
            username = tagged_user.get('username')
            if username in target_users:
                found_users.append(username)
        
        # Check mentions in caption
        for mention in post.get('mentions', []):
            if mention in target_users:
                found_users.append(mention)
        
        # Check coauthors
        for coauthor in post.get('coauthorProducers', []):
            username = coauthor.get('username')
            if username in target_users:
                found_users.append(username)
        
        # Return unique usernames
        return list(set(found_users))

    def _collect_download_tasks(self, post, relevant_users):
        """Collects all media URLs for a post and adds them to the download_tasks list."""
        post_id = post.get('id')
        post_type = post.get('type')
        
        for target_user in relevant_users:
            self.user_stats[target_user] += 1
            
            # Create user directory
            user_dir = Path(self.output_dir) / self._sanitize_filename(target_user)
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Helper to check existence and add task
            def add_task(file_type, url, file_suffix, user_dir_path):
                filename = f"{post_id}{file_suffix}"
                filepath = user_dir_path / filename
                
                if filepath.exists() and filepath.stat().st_size > 0:
                    self.stats['skipped'] += 1
                    return
                
                # Format: (file_type, url, filepath, filename, target_user)
                self.download_tasks.append((file_type, url, filepath, filename, target_user))

            # 1. Collect display image (thumbnail)
            if post.get('displayUrl'):
                add_task('image', post['displayUrl'], '_display.jpg', user_dir)
            
            # 2. Collect video if it's a video post
            if post_type == "Video" and post.get('videoUrl'):
                add_task('video', post['videoUrl'], '_video.mp4', user_dir)
            
            # 3. Collect additional images (for carousel posts)
            if post.get('images'):
                for img_idx, img_url in enumerate(post['images']):
                    add_task('image', img_url, f'_image_{img_idx+1}.jpg', user_dir)
            
            # 4. Collect child posts (carousel items)
            if post.get('childPosts'):
                for child_idx, child in enumerate(post['childPosts']):
                    if child.get('displayUrl'):
                        add_task('image', child['displayUrl'], f'_child_{child_idx+1}.jpg', user_dir)

    def run(self):
        """Executes the post processing and parallel downloading."""
        
        if not self.target_users:
            print("Error: Target users list is empty. Cannot process.")
            return

        # 1. Load JSON data
        print(f"Loading JSON from {self.json_file}...")
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.json_file}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_file}")
            return

        self.stats['total_posts'] = len(data)
        print(f"Found {len(data)} posts in JSON")
        print(f"Filtering for users: {', '.join(self.target_users)}\n")
        
        # Initialize user stats
        self.user_stats = {user: 0 for user in self.target_users}

        # Create base output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 2. Process each post and collect tasks
        for idx, post in enumerate(data, 1):
            owner_username = post.get('ownerUsername')
            post_type = post.get('type')
            
            # Find which target users are in this post
            relevant_users = self._get_target_users_in_post(post)
            
            if not relevant_users:
                continue
            
            self.stats['relevant_posts'] += 1
            
            print(f"[{idx}/{len(data)}] Post by @{owner_username} - Target users: {', '.join(relevant_users)} (Type: {post_type})")
            
            self._collect_download_tasks(post, relevant_users)

        # 3. Download files in parallel
        if self.download_tasks:
            total_tasks = len(self.download_tasks)
            print(f"\nStarting parallel download of {total_tasks} files using {self.workers} workers...")
            
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Map futures to their original task data
                future_to_task = {
                    executor.submit(self._download_file, task[1], task[2]): task 
                    for task in self.download_tasks
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    file_type, url, filepath, filename, target_user = task
                    
                    try:
                        success = future.result()
                        if success:
                            if file_type == 'video':
                                self.stats['videos_downloaded'] += 1
                            else:
                                self.stats['images_downloaded'] += 1
                        else:
                            self.stats['failed_downloads'] += 1
                    except Exception as e:
                        print(f" Critical Error with {filename}: {e}")
                        self.stats['failed_downloads'] += 1
        else:
            print("\nNo new media files found to download.")
        
        # 4. Print summary
        self._print_summary()

    def _print_summary(self):
        """Prints the final download summary."""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total posts in JSON: {self.stats['total_posts']}")
        print(f"Posts with target users: {self.stats['relevant_posts']}")
        print(f"\nPosts per target user:")
        for user, count in self.user_stats.items():
            print(f" @{user}: {count} posts")
        print(f"\nImages downloaded: {self.stats['images_downloaded']}")
        print(f"Videos downloaded: {self.stats['videos_downloaded']}")
        print(f"Files skipped (already exist): {self.stats['skipped']}")
        print(f"Failed downloads: {self.stats['failed_downloads']}")
        print(f"\nAll media saved to: {self.output_dir}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download Instagram media from Apify JSON, organized by target users (owner or tagged)'
    )
    parser.add_argument(
        'json_file',
        help='Path to JSON file from Apify'
    )
    parser.add_argument(
        '-o', '--output',
        default='instagram_downloads',
        help='Output directory for downloads (default: instagram_downloads)'
    )
    parser.add_argument(
        '-u', '--users',
        nargs='+',
        required=True,
        help='Target usernames to track (space-separated, REQUIRED)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel download workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    downloader = InstagramDownloader(
        json_file=args.json_file,
        output_dir=args.output,
        target_users=args.users,
        workers=args.workers
    )
    
    downloader.run()