import sqlite3
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str):
        """Initializes the database connection and creates tables if they don't exist."""
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS athletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER,
                contest_name TEXT,
                placing INTEGER,
                athlete_name TEXT,
                image_filename TEXT,
                local_path TEXT,
                UNIQUE(year, contest_name, athlete_name, image_filename)
            )
        ''')
        self.conn.commit()

    def insert_record(self, year, contest, placing, athlete, filename, local_path):
        """Inserts a new image record into the database, ignoring duplicates."""
        try:
            self.cursor.execute('''
                INSERT INTO athletes (year, contest_name, placing, athlete_name, image_filename, local_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (year, contest, placing, athlete, filename, str(local_path)))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass # Record already exists

    def is_athlete_processed(self, year, contest, athlete):
        """Check if we already downloaded images for this specific athlete."""
        self.cursor.execute('''
            SELECT 1 FROM athletes 
            WHERE year = ? AND contest_name = ? AND athlete_name = ?
        ''', (year, contest, athlete))
        return self.cursor.fetchone() is not None
        
    def close(self):
        """Safely close the database connection."""
        self.conn.close()