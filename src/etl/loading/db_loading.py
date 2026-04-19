import sqlite3
import threading
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Use check_same_thread=False to allow multi-threaded access with our own lock
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        with self.lock:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS athletes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER,
                    contest_name TEXT,
                    division TEXT,
                    placing INTEGER,
                    athlete_name TEXT,
                    image_filename TEXT,
                    UNIQUE(year, contest_name, division, athlete_name, image_filename)
                )
            ''')
            # Index for faster idempotency checks during scraping
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_athlete_lookup 
                ON athletes (year, contest_name, division, athlete_name)
            ''')
            self.conn.commit()

    def get_processed_athletes_for_contest(self, year, contest):
        """Returns a set of (division, athlete_name) already in DB for a contest."""
        with self.lock:
            self.cursor.execute('''
                SELECT DISTINCT division, athlete_name FROM athletes 
                WHERE year = ? AND contest_name = ?
            ''', (year, contest))
            return set(self.cursor.fetchall())

    def insert_record(self, year, contest, division, placing, athlete, filename, commit=True):
        with self.lock:
            try:
                self.cursor.execute('''
                    INSERT INTO athletes (year, contest_name, division, placing, athlete_name, image_filename)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (year, contest, division, placing, athlete, filename))
                if commit:
                    self.conn.commit()
            except sqlite3.IntegrityError:
                pass # Record already exists

    def is_athlete_processed(self, year, contest, division, athlete):
        with self.lock:
            self.cursor.execute('''
                SELECT 1 FROM athletes 
                WHERE year = ? AND contest_name = ? AND division = ? AND athlete_name = ?
            ''', (year, contest, division, athlete))
            return self.cursor.fetchone() is not None

    def commit(self):
        with self.lock:
            self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()