# IFBB Computer Vision Pipeline - Data Ingestion

This repository handles the extraction and ingestion of bodybuilding competition data from NPC News and other sources.

## Current Status
- **Extracted:** ~100,000 images (Years 2025-2026).
- **Storage:** ZIP files by year/competition on Google Drive.
- **Database:** SQLite tracking athletes, divisions, and image mappings.
- **Fixed:** Database locking issues during background Drive syncs (using safe snapshots).

## Roadmap
1. [ ] **Auto-generate Face DB and Full-body ReID DB**:
    - Process NPC News photos (pristine data).
    - Use Face Detection/Recognition to auto-crop.
    - Group by Athlete ID to create a specialized training set.
2. [ ] **Train Custom ReID Model**:
    - Focus on bodybuilding-specific features (tan, muscle definition, stage lighting).
3. [ ] **Fine-tune YOLO (yolo11s)**:
    - Handle messy edge cases (marshals, occlusion, overlapping athletes).
    - Hard negative mining on video frames.
4. [ ] **Real-time Pipeline**:
    - Integrate custom YOLO and ReID.
    - Optimize for CPU (Lightweight detectors + sparse ReID checks).

## Related Repositories
- `ifbb_athlete_tracking`: Tracking logic and pipeline.
- `ifbb_model_training`: Training scripts for YOLO and ReID.

## Notes
- NPC News data is considered "Ground Truth" for labels.
- Goal is to process 14 more years of data (~700k+ images expected).
