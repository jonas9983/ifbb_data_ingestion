import sqlite3
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str):
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
                division TEXT,
                placing INTEGER,
                athlete_name TEXT,
                image_filename TEXT,
                UNIQUE(year, contest_name, division, athlete_name, image_filename)
            )
        ''')
        self.conn.commit()

    def insert_record(self, year, contest, division, placing, athlete, filename):
        try:
            self.cursor.execute('''
                INSERT INTO athletes (year, contest_name, division, placing, athlete_name, image_filename)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (year, contest, division, placing, athlete, filename))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass # Record already exists

    def is_athlete_processed(self, year, contest, division, athlete):
        self.cursor.execute('''
            SELECT 1 FROM athletes 
            WHERE year = ? AND contest_name = ? AND division = ? AND athlete_name = ?
        ''', (year, contest, division, athlete))
        return self.cursor.fetchone() is not None
        
    def close(self):
        self.conn.close()