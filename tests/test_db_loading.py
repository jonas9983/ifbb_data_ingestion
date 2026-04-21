import pytest
import sqlite3
import os
from pathlib import Path
from ifbb_data_ingestion.etl.loading.db_loading import DatabaseManager

@pytest.fixture
def db_manager(tmp_path):
    db_file = tmp_path / "test_npc.db"
    manager = DatabaseManager(str(db_file))
    yield manager
    manager.close()

def test_create_tables(db_manager):
    # Check if table exists
    db_manager.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='athletes'")
    assert db_manager.cursor.fetchone() is not None

def test_insert_and_idempotency(db_manager):
    # Insert first record
    db_manager.insert_record(2025, "Olympia", "Classic Physique", 1, "Chris Bumstead", "cbum_1.jpg")
    
    # Check it exists
    assert db_manager.is_athlete_processed(2025, "Olympia", "Classic Physique", "Chris Bumstead") is True
    
    # Try to insert same record again (should not raise error due to IntegrityError handling)
    db_manager.insert_record(2025, "Olympia", "Classic Physique", 1, "Chris Bumstead", "cbum_1.jpg")
    
    # Count should still be 1
    db_manager.cursor.execute("SELECT COUNT(*) FROM athletes")
    assert db_manager.cursor.fetchone()[0] == 1

def test_get_processed_athletes(db_manager):
    db_manager.insert_record(2025, "Olympia", "Classic Physique", 1, "Chris Bumstead", "cbum_1.jpg")
    db_manager.insert_record(2025, "Olympia", "Men's Physique", 1, "Ryan Terry", "ryan_1.jpg")
    
    processed = db_manager.get_processed_athletes_for_contest(2025, "Olympia")
    assert len(processed) == 2
    assert ("Classic Physique", "Chris Bumstead") in processed
    assert ("Men's Physique", "Ryan Terry") in processed

def test_backup(db_manager, tmp_path):
    db_manager.insert_record(2025, "Olympia", "Classic Physique", 1, "Chris Bumstead", "cbum_1.jpg")
    
    backup_path = tmp_path / "backup.db"
    db_manager.backup(str(backup_path))
    
    assert backup_path.exists()
    
    # Verify backup content
    backup_conn = sqlite3.connect(str(backup_path))
    cursor = backup_conn.cursor()
    cursor.execute("SELECT athlete_name FROM athletes")
    assert cursor.fetchone()[0] == "Chris Bumstead"
    backup_conn.close()
