import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import tempfile
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io

# Import our OMR processing modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from omr_processor.image_preprocessor import ImagePreprocessor
from omr_processor.bubble_detector import BubbleDetector
from omr_processor.answer_extractor import AnswerExtractor
from omr_processor.sheet_classifier import SheetClassifier
from utils.database import DatabaseManager
from utils.helpers import (
    setup_logging, validate_image_file, format_processing_time,
    export_to_csv, export_to_excel
)
from utils.config import get_web_app_config, get_scoring_config

class OMRStreamlitApp:
    """
    Streamlit web application for OMR evaluation system.
    """
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.setup_page_config()
        self.initialize_components()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        config = get_web_app_config()
        
        st.set_page_config(
            page_title=config.get('title', 'OMR Evaluation System'),
            page_icon="📝",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        """Initialize processing components."""
        self.preprocessor = ImagePreprocessor()
        self.bubble_detector = BubbleDetector()
        self.answer_extractor = AnswerExtractor()
        self.sheet_classifier = SheetClassifier()
        self.db_manager = DatabaseManager()
        self.logger = setup_logging()
    
    def setup_session_state(self):
        """Setup Streamlit session state variables."""
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = []
        
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def run(self):
        """Run the main application."""
        st.title("🎯 OMR Evaluation System")
        st.markdown("### Automated Optical Mark Recognition for Educational Assessments")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Process OMR Sheets", "Results Dashboard", "Manage Answer Keys", "System Settings"]
        )
        
        if page == "Process OMR Sheets":
            self.show_processing_page()
        elif page == "Results Dashboard":
            self.show_dashboard_page()
        elif page == "Manage Answer Keys":
            self.show_answer_keys_page()
        elif page == "System Settings":
            self.show_settings_page()
    
    def show_processing_page(self):
        """Show the main processing page."""
        st.header("📋 Process OMR Sheets")
        
        # File upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload OMR Sheet Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload one or more OMR sheet images for processing"
            )
        
        with col2:
            st.markdown("### Processing Options")
            
            # Sheet version selection
            sheet_versions = ['AUTO_DETECT', 'SET_A', 'SET_B', 'SET_C', 'SET_D']
            selected_version = st.selectbox(
                "Sheet Version",
                sheet_versions,
                help="Select sheet version or use auto-detection"
            )
            
            # Processing options
            save_to_db = st.checkbox("Save results to database", value=True)
            show_debug = st.checkbox("Show debug information", value=False)
        
        if uploaded_files:
            st.markdown("---")
            
            # Process button
            if st.button("🚀 Process OMR Sheets", type="primary"):
                self.process_uploaded_files(uploaded_files, selected_version, save_to_db, show_debug)
        
        # Show processing history
        if st.session_state.processing_results:
            st.markdown("---")
            st.subheader("Recent Processing Results")
            self.show_processing_history()
    
    def process_uploaded_files(self, uploaded_files: List, sheet_version: str, 
                              save_to_db: bool, show_debug: bool):
        """
        Process uploaded OMR sheet files.
        
        Args:
            uploaded_files: List of uploaded files
            sheet_version: Selected sheet version
            save_to_db: Whether to save results to database
            show_debug: Whether to show debug information
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process the file
                result = self.process_single_file(
                    tmp_path, uploaded_file.name, sheet_version, show_debug
                )
                
                if result['success']:
                    results.append(result)
                    
                    # Save to database if requested
                    if save_to_db:
                        self.db_manager.save_processing_result(result)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        # Update session state
        st.session_state.processing_results.extend(results)
        
        # Show results
        if results:
            st.success(f"Successfully processed {len(results)} out of {len(uploaded_files)} files!")
            self.show_batch_results(results)
        else:
            st.error("No files were processed successfully.")
        
        progress_bar.empty()
        status_text.empty()
    
    def process_single_file(self, image_path: str, filename: str, 
                           sheet_version: str, show_debug: bool) -> Dict[str, Any]:
        """
        Process a single OMR sheet file.
        
        Args:
            image_path: Path to the image file
            filename: Original filename
            sheet_version: Sheet version
            show_debug: Whether to show debug info
            
        Returns:
            Processing result dictionary
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Preprocess image
            preprocessed_image = self.preprocessor.preprocess_image(image_path)
            
            # Step 2: Classify sheet
            if sheet_version == 'AUTO_DETECT':
                classification_result = self.sheet_classifier.classify_sheet(preprocessed_image)
                detected_version = classification_result.get('detected_set', 'SET_A')
            else:
                detected_version = sheet_version
                classification_result = {'detected_set': sheet_version, 'confidence': 1.0}
            
            # Step 3: Detect bubbles
            bubble_grid = self.bubble_detector.detect_answer_bubbles_grid(preprocessed_image)
            
            if not bubble_grid['success']:
                return {
                    'success': False,
                    'error': 'Failed to detect bubble grid',
                    'filename': filename
                }
            
            # Step 4: Extract answers
            answer_key = self.sheet_classifier.get_answer_key_for_set(detected_version)
            extraction_result = self.answer_extractor.extract_answers(
                preprocessed_image, bubble_grid, answer_key
            )
            
            if not extraction_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to extract answers',
                    'filename': filename
                }
            
            # Step 5: Generate detailed report
            student_info = self.extract_student_info(filename)
            detailed_report = self.answer_extractor.generate_detailed_report(
                extraction_result, student_info
            )
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create final result
            result = {
                'success': True,
                'timestamp': start_time,
                'filename': filename,
                'image_filename': filename,
                'sheet_version': detected_version,
                'student_info': student_info,
                'classification_result': classification_result,
                'bubble_detection_result': bubble_grid,
                'extraction_result': extraction_result,
                'detailed_report': detailed_report,
                'processing_time_seconds': processing_time,
                'status': 'completed'
            }
            
            # Add score summary to main result
            if detailed_report.get('overall_score'):
                result.update({
                    'score_summary': detailed_report['overall_score'],
                    'subject_breakdown': detailed_report.get('subject_breakdown', [])
                })
            
            if show_debug:
                result['debug_info'] = {
                    'preprocessed_image_shape': preprocessed_image.shape,
                    'bubbles_detected': bubble_grid.get('total_bubbles', 0),
                    'processing_stats': extraction_result.get('processing_stats', {})
                }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'timestamp': start_time
            }
    
    def extract_student_info(self, filename: str) -> Dict[str, str]:
        """
        Extract student information from filename or other sources.
        
        Args:
            filename: Image filename
            
        Returns:
            Student information dictionary
        """
        # Simple extraction from filename - in practice, this could be more sophisticated
        base_name = os.path.splitext(filename)[0]
        
        return {
            'id': base_name.split('_')[0] if '_' in base_name else 'UNKNOWN',
            'name': base_name.replace('_', ' ').title(),
            'filename': filename
        }
    
    def show_batch_results(self, results: List[Dict[str, Any]]):
        """Show results from batch processing."""
        st.subheader("📊 Processing Results Summary")
        
        # Summary metrics
        total_processed = len(results)
        successful = len([r for r in results if r.get('success', False)])
        avg_score = np.mean([
            r.get('score_summary', {}).get('accuracy_percentage', 0) 
            for r in results if r.get('success', False)
        ]) if successful > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", total_processed)
        col2.metric("Successful", successful)
        col3.metric("Average Score", f"{avg_score:.1f}%")
        
        # Detailed results table
        if successful > 0:
            results_data = []
            for result in results:
                if result.get('success'):
                    score_summary = result.get('score_summary', {})
                    results_data.append({
                        'Student ID': result.get('student_info', {}).get('id', 'N/A'),
                        'Student Name': result.get('student_info', {}).get('name', 'N/A'),
                        'Sheet Version': result.get('sheet_version', 'N/A'),
                        'Score': score_summary.get('raw_score', 'N/A'),
                        'Accuracy': f"{score_summary.get('accuracy_percentage', 0):.1f}%",
                        'Processing Time': f"{result.get('processing_time_seconds', 0):.2f}s"
                    })
            
            if results_data:
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create Excel file in memory
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Results', index=False)
                    
                    st.download_button(
                        "📊 Download Excel",
                        excel_buffer.getvalue(),
                        file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    def show_processing_history(self):
        """Show recent processing history."""
        recent_results = st.session_state.processing_results[-10:]  # Show last 10
        
        for i, result in enumerate(reversed(recent_results)):
            with st.expander(f"📄 {result['filename']} - {result.get('student_info', {}).get('name', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- Student ID: {result.get('student_info', {}).get('id', 'N/A')}")
                    st.write(f"- Sheet Version: {result.get('sheet_version', 'N/A')}")
                    st.write(f"- Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
                
                with col2:
                    if result.get('score_summary'):
                        score = result['score_summary']
                        st.write("**Score Summary:**")
                        st.write(f"- Score: {score.get('raw_score', 'N/A')}")
                        st.write(f"- Accuracy: {score.get('accuracy_percentage', 0):.1f}%")
                        st.write(f"- Attempted: {score.get('attempted_questions', 0)}")
    
    def show_dashboard_page(self):
        """Show the results dashboard page."""
        st.header("📈 Results Dashboard")
        
        # Get statistics from database
        stats = self.db_manager.get_statistics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Results", stats['total_results'])
        col2.metric("Average Accuracy", f"{stats['average_accuracy']:.1f}%")
        col3.metric("Today's Results", len([r for r in stats['daily_results'] if r['date'] == datetime.now().strftime('%Y-%m-%d')]))
        col4.metric("Processing Time", f"{stats['processing_time_stats'].get('avg_time', 0):.2f}s")
        
        # Charts
        if stats['daily_results']:
            st.subheader("📊 Daily Processing Volume")
            
            df_daily = pd.DataFrame(stats['daily_results'])
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            
            fig = px.line(df_daily, x='date', y='count', title='Daily Processing Results')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top students
        if stats['top_students']:
            st.subheader("🏆 Top Performing Students")
            
            df_top = pd.DataFrame(stats['top_students'])
            fig = px.bar(df_top, x='student_name', y='avg_score', 
                        title='Top Students by Average Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent results table
        st.subheader("📋 Recent Results")
        recent_results = self.db_manager.get_processing_results(limit=20)
        
        if recent_results:
            results_data = []
            for result in recent_results:
                results_data.append({
                    'ID': result['id'],
                    'Timestamp': result['timestamp'],
                    'Student': result['student_name'],
                    'Score': f"{result['total_score']}/{result['total_questions']}",
                    'Accuracy': f"{result['accuracy_percentage']:.1f}%",
                    'Version': result['sheet_version']
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
    
    def show_answer_keys_page(self):
        """Show the answer keys management page."""
        st.header("🔑 Manage Answer Keys")
        
        # Get existing answer keys
        answer_keys = self.db_manager.get_all_answer_keys()
        
        # Display existing keys
        if answer_keys:
            st.subheader("📚 Existing Answer Keys")
            
            for set_version, key_data in answer_keys.items():
                with st.expander(f"Answer Key: {set_version}"):
                    st.write(f"Total Questions: {len(key_data)}")
                    
                    # Show first 10 answers as sample
                    sample_data = {f"Q{k}": v for k, v in list(key_data.items())[:10]}
                    st.json(sample_data)
                    
                    if st.button(f"Delete {set_version}", key=f"delete_{set_version}"):
                        # In a real implementation, add delete functionality
                        st.warning("Delete functionality would be implemented here")
        
        # Add new answer key
        st.subheader("➕ Add New Answer Key")
        
        new_set_version = st.selectbox(
            "Set Version",
            ['SET_A', 'SET_B', 'SET_C', 'SET_D', 'CUSTOM']
        )
        
        if new_set_version == 'CUSTOM':
            custom_version = st.text_input("Enter custom set version:")
            if custom_version:
                new_set_version = custom_version
        
        # Upload answer key file
        uploaded_key_file = st.file_uploader(
            "Upload Answer Key (JSON format)",
            type=['json'],
            help="Upload a JSON file with question numbers as keys and answers (A/B/C/D) as values"
        )
        
        if uploaded_key_file and st.button("💾 Save Answer Key"):
            try:
                key_data = json.loads(uploaded_key_file.getvalue())
                # Convert string keys to integers
                converted_key = {int(k): v for k, v in key_data.items()}
                
                if self.db_manager.save_answer_key(new_set_version, converted_key):
                    st.success(f"Answer key for {new_set_version} saved successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to save answer key")
            except Exception as e:
                st.error(f"Error processing answer key file: {str(e)}")
    
    def show_settings_page(self):
        """Show the system settings page."""
        st.header("⚙️ System Settings")
        
        # Database settings
        st.subheader("🗄️ Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clean Old Results"):
                days_to_keep = st.number_input("Days to keep", min_value=1, value=90)
                if st.button("Confirm Cleanup"):
                    deleted = self.db_manager.cleanup_old_results(days_to_keep)
                    st.success(f"Deleted {deleted} old records")
        
        with col2:
            if st.button("💾 Backup Database"):
                backup_path = f"backup/omr_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                if self.db_manager.backup_database(backup_path):
                    st.success(f"Database backed up to {backup_path}")
                else:
                    st.error("Backup failed")
        
        # Export options
        st.subheader("📤 Export Data")
        
        date_range = st.date_input(
            "Select date range for export",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            help="Select start and end dates for data export"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export to CSV"):
                    export_path = f"exports/omr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    if self.db_manager.export_results_to_csv(
                        export_path, 
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time())
                    ):
                        st.success(f"Data exported to {export_path}")
                    else:
                        st.error("Export failed")
            
            with col2:
                st.info("Excel export functionality would be implemented here")
        
        # System information
        st.subheader("ℹ️ System Information")
        
        from utils.helpers import get_system_info, memory_usage_mb
        
        system_info = get_system_info()
        memory_usage = memory_usage_mb()
        
        st.json({
            "System": system_info['system'],
            "Python Version": system_info['python_version'].split()[0],
            "Memory Usage": f"{memory_usage:.2f} MB",
            "Database Path": self.db_manager.db_path,
            "Total Records": self.db_manager.get_statistics()['total_results']
        })

def main():
    """Main function to run the Streamlit app."""
    app = OMRStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()