import json
import os
import requests
from pathlib import Path
from urllib.parse import urlparse
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False

def get_target_users_in_post(post, target_users):
    """
    Get list of target users that appear in this post (as owner or tagged)
    
    Args:
        post: Instagram post dict
        target_users: Set of target usernames
    
    Returns:
        List of target usernames found in this post
    """
    found_users = []
    
    # Check if owner is a target user
    owner = post.get('ownerUsername')
    if owner in target_users:
        found_users.append(owner)
    
    # Check tagged users
    tagged = post.get('taggedUsers', [])
    for tagged_user in tagged:
        username = tagged_user.get('username')
        if username in target_users:
            found_users.append(username)
    
    # Check mentions in caption
    mentions = post.get('mentions', [])
    for mention in mentions:
        if mention in target_users:
            found_users.append(mention)
    
    # Check coauthors
    coauthors = post.get('coauthorProducers', [])
    for coauthor in coauthors:
        username = coauthor.get('username')
        if username in target_users:
            found_users.append(username)
    
    # Return unique usernames
    return list(set(found_users))

def process_instagram_json(json_file, output_dir="downloads", target_users=None, workers=4):
    """
    Process Instagram JSON and download media organized by target users
    
    Args:
        json_file: Path to JSON file from Apify
        output_dir: Base directory for downloads
        target_users: List of usernames to track (downloads posts where they appear)
        workers: Number of parallel download threads
    """
    
    # Load JSON data
    print(f"Loading JSON from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} posts in JSON")
    
    if target_users:
        target_users_set = set(target_users)
        print(f"Filtering for users: {', '.join(target_users)}\n")
    else:
        print("Error: You must specify target users with -u flag")
        return
    
    # Create base output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_posts': len(data),
        'relevant_posts': 0,
        'images_downloaded': 0,
        'videos_downloaded': 0,
        'failed_downloads': 0,
        'skipped': 0
    }
    
    # Track downloads per user
    user_stats = {user: 0 for user in target_users}
    
    # Collect all download tasks
    download_tasks = []
    
    # Process each post
    for idx, post in enumerate(data, 1):
        post_id = post.get('id')
        post_type = post.get('type')
        owner_username = post.get('ownerUsername')
        
        # Find which target users are in this post
        relevant_users = get_target_users_in_post(post, target_users_set)
        
        if not relevant_users:
            continue
        
        stats['relevant_posts'] += 1
        
        print(f"[{idx}/{len(data)}] Post by @{owner_username} - Target users: {', '.join(relevant_users)} (Type: {post_type})")
        
        # Download media to each relevant user's folder
        for target_user in relevant_users:
            user_stats[target_user] += 1
            
            # Create user directory
            user_dir = Path(output_dir) / sanitize_filename(target_user)
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect display image (thumbnail)
            if post.get('displayUrl'):
                img_filename = f"{post_id}_display.jpg"
                img_path = user_dir / img_filename
                
                if img_path.exists() and img_path.stat().st_size > 0:
                    stats['skipped'] += 1
                else:
                    download_tasks.append(('image', post['displayUrl'], img_path, img_filename, target_user))
            
            # Collect video if it's a video post
            if post_type == "Video" and post.get('videoUrl'):
                video_filename = f"{post_id}_video.mp4"
                video_path = user_dir / video_filename
                
                if video_path.exists() and video_path.stat().st_size > 0:
                    stats['skipped'] += 1
                else:
                    download_tasks.append(('video', post['videoUrl'], video_path, video_filename, target_user))
            
            # Collect additional images (for carousel posts)
            if post.get('images'):
                for img_idx, img_url in enumerate(post['images']):
                    img_filename = f"{post_id}_image_{img_idx+1}.jpg"
                    img_path = user_dir / img_filename
                    
                    if img_path.exists() and img_path.stat().st_size > 0:
                        stats['skipped'] += 1
                    else:
                        download_tasks.append(('image', img_url, img_path, img_filename, target_user))
            
            # Collect child posts (carousel items)
            if post.get('childPosts'):
                for child_idx, child in enumerate(post['childPosts']):
                    if child.get('displayUrl'):
                        child_filename = f"{post_id}_child_{child_idx+1}.jpg"
                        child_path = user_dir / child_filename
                        
                        if child_path.exists() and child_path.stat().st_size > 0:
                            stats['skipped'] += 1
                        else:
                            download_tasks.append(('image', child['displayUrl'], child_path, child_filename, target_user))
    
    # Download files in parallel
    if download_tasks:
        print(f"\nStarting parallel download of {len(download_tasks)} files using {workers} workers...")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(download_file, task[1], task[2]): task 
                for task in download_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                file_type, url, filepath, filename, target_user = task
                
                try:
                    success = future.result()
                    if success:
                        if file_type == 'video':
                            stats['videos_downloaded'] += 1
                        else:
                            stats['images_downloaded'] += 1
                    else:
                        stats['failed_downloads'] += 1
                except Exception as e:
                    print(f"  Error with {filename}: {e}")
                    stats['failed_downloads'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total posts in JSON: {stats['total_posts']}")
    print(f"Posts with target users: {stats['relevant_posts']}")
    print(f"\nPosts per target user:")
    for user, count in user_stats.items():
        print(f"  @{user}: {count} posts")
    print(f"\nImages downloaded: {stats['images_downloaded']}")
    print(f"Videos downloaded: {stats['videos_downloaded']}")
    print(f"Files skipped (already exist): {stats['skipped']}")
    print(f"Failed downloads: {stats['failed_downloads']}")
    print(f"\nAll media saved to: {output_dir}")
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
    
    process_instagram_json(
        args.json_file,
        args.output,
        args.users,
        args.workers
    )