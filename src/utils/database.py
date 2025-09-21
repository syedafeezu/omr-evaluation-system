import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
import os
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Data class for processing results."""
    id: Optional[int]
    timestamp: datetime
    student_id: str
    student_name: str
    image_filename: str
    sheet_version: str
    total_score: int
    total_questions: int
    accuracy_percentage: float
    processing_time_seconds: float
    results_json: str
    status: str

class DatabaseManager:
    """
    Manages SQLite database operations for the OMR evaluation system.
    """
    
    def __init__(self, db_path: str = "data/omr_results.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database_directory()
        self._initialize_database()
    
    def _ensure_database_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _initialize_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create processing_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    student_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    image_filename TEXT NOT NULL,
                    sheet_version TEXT,
                    total_score INTEGER NOT NULL,
                    total_questions INTEGER NOT NULL,
                    accuracy_percentage REAL NOT NULL,
                    processing_time_seconds REAL NOT NULL,
                    results_json TEXT NOT NULL,
                    status TEXT DEFAULT 'completed',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create answer_keys table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS answer_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_version TEXT NOT NULL UNIQUE,
                    answer_key_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create processing_sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    exam_name TEXT,
                    exam_date DATE,
                    total_sheets INTEGER DEFAULT 0,
                    processed_sheets INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Create subject_scores table for detailed breakdown
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS subject_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER,
                    subject_name TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    total_questions INTEGER NOT NULL,
                    percentage REAL NOT NULL,
                    FOREIGN KEY (result_id) REFERENCES processing_results (id)
                        ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_processing_results_student_id 
                ON processing_results(student_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_processing_results_timestamp 
                ON processing_results(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_subject_scores_result_id 
                ON subject_scores(result_id)
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def save_processing_result(self, result: Dict[str, Any]) -> int:
        """
        Save processing result to database.
        
        Args:
            result: Processing result dictionary
            
        Returns:
            ID of inserted record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract basic information
            student_info = result.get('student_info', {})
            score_summary = result.get('score_summary', {})
            
            # Insert main result
            cursor.execute('''
                INSERT INTO processing_results 
                (timestamp, student_id, student_name, image_filename, sheet_version,
                 total_score, total_questions, accuracy_percentage, 
                 processing_time_seconds, results_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.get('timestamp', datetime.now()),
                student_info.get('id', ''),
                student_info.get('name', ''),
                result.get('image_filename', ''),
                result.get('sheet_version', ''),
                score_summary.get('correct_answers', 0),
                score_summary.get('total_questions', 0),
                score_summary.get('accuracy_percentage', 0.0),
                result.get('processing_time_seconds', 0.0),
                json.dumps(result),
                result.get('status', 'completed')
            ))
            
            result_id = cursor.lastrowid
            
            # Save subject scores if available
            subject_breakdown = result.get('subject_breakdown', [])
            for subject in subject_breakdown:
                cursor.execute('''
                    INSERT INTO subject_scores 
                    (result_id, subject_name, score, total_questions, percentage)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    result_id,
                    subject['subject'],
                    subject['correct'],
                    subject['questions'],
                    subject['percentage']
                ))
            
            conn.commit()
            return result_id
    
    def get_processing_result(self, result_id: int) -> Optional[Dict[str, Any]]:
        """
        Get processing result by ID.
        
        Args:
            result_id: Result ID
            
        Returns:
            Processing result or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM processing_results WHERE id = ?
            ''', (result_id,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['results_json'] = json.loads(result['results_json'])
                
                # Get subject scores
                cursor.execute('''
                    SELECT * FROM subject_scores WHERE result_id = ?
                ''', (result_id,))
                
                subject_rows = cursor.fetchall()
                result['subject_scores'] = [dict(row) for row in subject_rows]
                
                return result
        
        return None
    
    def get_processing_results(self, 
                             limit: int = 100, 
                             offset: int = 0,
                             student_id: str = None,
                             date_from: datetime = None,
                             date_to: datetime = None) -> List[Dict[str, Any]]:
        """
        Get processing results with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            student_id: Filter by student ID
            date_from: Filter by start date
            date_to: Filter by end date
            
        Returns:
            List of processing results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM processing_results 
                WHERE 1=1
            '''
            params = []
            
            if student_id:
                query += ' AND student_id = ?'
                params.append(student_id)
            
            if date_from:
                query += ' AND timestamp >= ?'
                params.append(date_from)
            
            if date_to:
                query += ' AND timestamp <= ?'
                params.append(date_to)
            
            query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['results_json'] = json.loads(result['results_json'])
                results.append(result)
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total results
            cursor.execute('SELECT COUNT(*) as count FROM processing_results')
            total_results = cursor.fetchone()['count']
            
            # Average accuracy
            cursor.execute('''
                SELECT AVG(accuracy_percentage) as avg_accuracy 
                FROM processing_results
            ''')
            avg_accuracy = cursor.fetchone()['avg_accuracy'] or 0.0
            
            # Results by day (last 30 days)
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM processing_results 
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''')
            daily_results = [dict(row) for row in cursor.fetchall()]
            
            # Top performing students
            cursor.execute('''
                SELECT student_name, student_id, AVG(accuracy_percentage) as avg_score
                FROM processing_results
                GROUP BY student_id, student_name
                HAVING COUNT(*) >= 1
                ORDER BY avg_score DESC
                LIMIT 10
            ''')
            top_students = [dict(row) for row in cursor.fetchall()]
            
            # Processing time statistics
            cursor.execute('''
                SELECT 
                    AVG(processing_time_seconds) as avg_time,
                    MIN(processing_time_seconds) as min_time,
                    MAX(processing_time_seconds) as max_time
                FROM processing_results
            ''')
            time_stats = dict(cursor.fetchone())
            
            return {
                'total_results': total_results,
                'average_accuracy': round(avg_accuracy, 2),
                'daily_results': daily_results,
                'top_students': top_students,
                'processing_time_stats': time_stats
            }
    
    def save_answer_key(self, set_version: str, answer_key: Dict[int, str]) -> bool:
        """
        Save answer key for a specific set version.
        
        Args:
            set_version: Set version (e.g., 'SET_A')
            answer_key: Answer key dictionary
            
        Returns:
            Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO answer_keys 
                    (set_version, answer_key_json, updated_at)
                    VALUES (?, ?, ?)
                ''', (
                    set_version,
                    json.dumps(answer_key),
                    datetime.now()
                ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving answer key: {e}")
            return False
    
    def get_answer_key(self, set_version: str) -> Optional[Dict[int, str]]:
        """
        Get answer key for a specific set version.
        
        Args:
            set_version: Set version
            
        Returns:
            Answer key dictionary or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT answer_key_json FROM answer_keys 
                WHERE set_version = ?
            ''', (set_version,))
            
            row = cursor.fetchone()
            if row:
                answer_key = json.loads(row['answer_key_json'])
                # Convert string keys back to integers
                return {int(k): v for k, v in answer_key.items()}
        
        return None
    
    def get_all_answer_keys(self) -> Dict[str, Dict[int, str]]:
        """
        Get all available answer keys.
        
        Returns:
            Dictionary of all answer keys
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT set_version, answer_key_json FROM answer_keys
            ''')
            
            answer_keys = {}
            for row in cursor.fetchall():
                answer_key = json.loads(row['answer_key_json'])
                answer_keys[row['set_version']] = {int(k): v for k, v in answer_key.items()}
            
            return answer_keys
    
    def export_results_to_csv(self, output_path: str, 
                             date_from: datetime = None,
                             date_to: datetime = None) -> bool:
        """
        Export results to CSV file.
        
        Args:
            output_path: Output CSV file path
            date_from: Optional start date filter
            date_to: Optional end date filter
            
        Returns:
            Success status
        """
        try:
            import pandas as pd
            
            results = self.get_processing_results(
                limit=10000,  # Large limit for export
                date_from=date_from,
                date_to=date_to
            )
            
            if not results:
                return False
            
            # Flatten results for CSV
            flattened_results = []
            for result in results:
                base_data = {
                    'id': result['id'],
                    'timestamp': result['timestamp'],
                    'student_id': result['student_id'],
                    'student_name': result['student_name'],
                    'image_filename': result['image_filename'],
                    'sheet_version': result['sheet_version'],
                    'total_score': result['total_score'],
                    'total_questions': result['total_questions'],
                    'accuracy_percentage': result['accuracy_percentage'],
                    'processing_time_seconds': result['processing_time_seconds'],
                    'status': result['status']
                }
                flattened_results.append(base_data)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def cleanup_old_results(self, days_to_keep: int = 90) -> int:
        """
        Clean up old processing results.
        
        Args:
            days_to_keep: Number of days to keep results
            
        Returns:
            Number of deleted records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM processing_results 
                WHERE timestamp < date('now', '-{} days')
            '''.format(days_to_keep))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create backup of the database.
        
        Args:
            backup_path: Backup file path
            
        Returns:
            Success status
        """
        try:
            import shutil
            
            # Ensure backup directory exists
            backup_dir = os.path.dirname(backup_path)
            if backup_dir and not os.path.exists(backup_dir):
                os.makedirs(backup_dir, exist_ok=True)
            
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            print(f"Error creating database backup: {e}")
            return False