# IFBB Data Ingestion Pipeline

A robust ETL pipeline for extracting, transforming, and loading IFBB (International Federation of Bodybuilding) athlete data and contest results.

## Recent Updates
- **Package Restructuring:** Now organized as a proper Python package `ifbb_data_ingestion`.
- **Stealth Scraper:** Enhanced NPC News scraper with custom User-Agents, random delays, and improved timeout handling to bypass rate-limiting.
- **Smart Skipping:** The scraper now checks the local database and skips contest navigation entirely if the data already exists, reducing network traffic by ~90% on re-runs.
- **Async Drive Uploads:** Contest results are zipped and uploaded to Google Drive asynchronously in the background.

## Project Structure
- `src/ifbb_data_ingestion/`: Core package containing ETL logic.
- `services/`: Service-specific entry points (e.g., NPC News scraper).
- `scripts/`: Utility scripts for manual uploads and data migration.
- `data/`: Local storage for images, database, and staging.

## Installation
Ensure you have [Rclone](https://rclone.org/) installed and configured for Google Drive.

1. Clone the repository and install the package in editable mode:
   ```bash
   pip install -e .
   ```

2. Install Playwright browsers:
   ```bash
   playwright install chromium
   ```

## Usage

### NPC News Scraper
Run the scraper for specific years. It will automatically download the latest DB from Drive, scrape new data, and sync zips back to Drive.
```bash
python3 -m services.npc_news.main --years 2024 2025
```

### Manual Upload
If you have data in `data/upload_staging` that needs to be synced manually:
```bash
python3 scripts/manual_upload.py --folder YOUR_FOLDER_NAME
```

### Migration
To migrate files between Drive locations and bundle them into zips:
```bash
python3 scripts/migrate_to_zip.py
```

## Configuration
Settings for disk limits, concurrent workers, and Drive paths can be found in `services/npc_news/main.py` under the `CONFIG` dictionary.

