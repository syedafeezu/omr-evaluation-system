import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from dataclasses import dataclass

@dataclass
class AnswerChoice:
    """Represents a single answer choice (A, B, C, D)"""
    option: str
    is_selected: bool
    confidence: float
    bubble_info: Dict[str, Any]

@dataclass
class QuestionResult:
    """Represents the result for a single question"""
    question_number: int
    selected_answer: Optional[str]
    confidence: float
    all_choices: List[AnswerChoice]
    is_multiple_selection: bool = False
    is_ambiguous: bool = False

class AnswerExtractor:
    """
    Extracts answers from detected bubbles in OMR sheets.
    """
    
    def __init__(self, fill_threshold: float = 0.3, confidence_threshold: float = 0.7):
        """
        Initialize answer extractor.
        
        Args:
            fill_threshold: Minimum fill ratio to consider a bubble as selected
            confidence_threshold: Minimum confidence to accept an answer
        """
        self.fill_threshold = fill_threshold
        self.confidence_threshold = confidence_threshold
        self.option_labels = ['A', 'B', 'C', 'D', 'E']  # Support up to 5 options
    
    def extract_answers(self, image: np.ndarray, bubble_grid: Dict[str, Any], 
                       answer_key: Dict[int, str] = None) -> Dict[str, Any]:
        """
        Extract answers from the bubble grid.
        
        Args:
            image: Preprocessed grayscale image
            bubble_grid: Organized bubble grid from BubbleDetector
            answer_key: Optional answer key for scoring
            
        Returns:
            Dictionary with extracted answers and scores
        """
        if not bubble_grid.get('success', False):
            return {
                'success': False,
                'error': 'Invalid bubble grid',
                'raw_results': [],
                'score_summary': None
            }
        
        bubble_rows = bubble_grid['bubble_rows']
        results = []
        
        # Process each row (question)
        for question_num, row_bubbles in enumerate(bubble_rows, 1):
            question_result = self._process_question_row(
                image, row_bubbles, question_num
            )
            results.append(question_result)
        
        # Calculate scores if answer key provided
        score_summary = None
        if answer_key:
            score_summary = self._calculate_scores(results, answer_key)
        
        return {
            'success': True,
            'total_questions': len(results),
            'results': results,
            'score_summary': score_summary,
            'processing_stats': self._get_processing_stats(results)
        }
    
    def _process_question_row(self, image: np.ndarray, row_bubbles: List[Dict[str, Any]], 
                             question_num: int) -> QuestionResult:
        """
        Process a single question row to extract the answer.
        
        Args:
            image: Grayscale image
            row_bubbles: List of bubbles in this row
            question_num: Question number
            
        Returns:
            QuestionResult object
        """
        choices = []
        
        # Analyze each bubble in the row
        for i, bubble in enumerate(row_bubbles):
            if i >= len(self.option_labels):
                break  # Skip extra bubbles
                
            option_letter = self.option_labels[i]
            is_filled, fill_ratio = self._analyze_bubble_fill(image, bubble)
            confidence = min(fill_ratio * 2, 1.0)  # Scale confidence
            
            choice = AnswerChoice(
                option=option_letter,
                is_selected=is_filled,
                confidence=confidence,
                bubble_info=bubble
            )
            choices.append(choice)
        
        # Determine the selected answer
        selected_choices = [c for c in choices if c.is_selected]
        
        # Handle different scenarios
        if len(selected_choices) == 0:
            # No answer selected
            return QuestionResult(
                question_number=question_num,
                selected_answer=None,
                confidence=0.0,
                all_choices=choices
            )
        elif len(selected_choices) == 1:
            # Single answer selected
            selected = selected_choices[0]
            return QuestionResult(
                question_number=question_num,
                selected_answer=selected.option,
                confidence=selected.confidence,
                all_choices=choices
            )
        else:
            # Multiple answers selected - take the one with highest confidence
            best_choice = max(selected_choices, key=lambda c: c.confidence)
            return QuestionResult(
                question_number=question_num,
                selected_answer=best_choice.option,
                confidence=best_choice.confidence,
                all_choices=choices,
                is_multiple_selection=True,
                is_ambiguous=True
            )
    
    def _analyze_bubble_fill(self, image: np.ndarray, bubble: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Analyze whether a bubble is filled.
        
        Args:
            image: Grayscale image
            bubble: Bubble information
            
        Returns:
            Tuple of (is_filled, fill_ratio)
        """
        # Extract bubble region
        x, y, w, h = bubble['bounding_box']
        
        # Expand region slightly for better analysis
        padding = 2
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        roi = image[y1:y2, x1:x2]
        
        # Create circular mask
        center_x = (x2 - x1) // 2
        center_y = (y2 - y1) // 2
        radius = min(w, h) // 2 - 2
        
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Apply mask
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        # Calculate fill statistics
        mask_pixels = mask > 0
        roi_values = masked_roi[mask_pixels]
        
        if len(roi_values) == 0:
            return False, 0.0
        
        # Calculate threshold based on local statistics
        roi_mean = np.mean(roi_values)
        roi_std = np.std(roi_values)
        local_threshold = max(roi_mean - 0.5 * roi_std, 50)  # Adaptive threshold
        
        # Count dark pixels
        dark_pixels = np.sum(roi_values < local_threshold)
        total_pixels = len(roi_values)
        
        fill_ratio = dark_pixels / total_pixels
        is_filled = fill_ratio >= self.fill_threshold
        
        return is_filled, fill_ratio
    
    def _calculate_scores(self, results: List[QuestionResult], 
                         answer_key: Dict[int, str]) -> Dict[str, Any]:
        """
        Calculate scores based on answer key.
        
        Args:
            results: List of question results
            answer_key: Correct answers
            
        Returns:
            Score summary dictionary
        """
        total_questions = len(results)
        correct_count = 0
        attempted_count = 0
        ambiguous_count = 0
        multiple_selection_count = 0
        
        question_scores = []
        
        for result in results:
            attempted = result.selected_answer is not None
            if attempted:
                attempted_count += 1
            
            if result.is_ambiguous:
                ambiguous_count += 1
            
            if result.is_multiple_selection:
                multiple_selection_count += 1
            
            # Check if answer is correct
            correct_answer = answer_key.get(result.question_number)
            is_correct = (attempted and 
                         result.selected_answer == correct_answer and 
                         not result.is_ambiguous)
            
            if is_correct:
                correct_count += 1
            
            question_scores.append({
                'question': result.question_number,
                'selected': result.selected_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'confidence': result.confidence,
                'attempted': attempted,
                'ambiguous': result.is_ambiguous
            })
        
        # Calculate percentages
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
        attempt_rate = (attempted_count / total_questions * 100) if total_questions > 0 else 0
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_count,
            'attempted_questions': attempted_count,
            'ambiguous_answers': ambiguous_count,
            'multiple_selections': multiple_selection_count,
            'accuracy_percentage': round(accuracy, 2),
            'attempt_rate_percentage': round(attempt_rate, 2),
            'raw_score': f"{correct_count}/{total_questions}",
            'question_scores': question_scores
        }
    
    def _get_processing_stats(self, results: List[QuestionResult]) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Args:
            results: List of question results
            
        Returns:
            Processing statistics
        """
        total_questions = len(results)
        high_confidence = sum(1 for r in results if r.confidence >= self.confidence_threshold)
        ambiguous = sum(1 for r in results if r.is_ambiguous)
        multiple_selection = sum(1 for r in results if r.is_multiple_selection)
        unanswered = sum(1 for r in results if r.selected_answer is None)
        
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0
        
        return {
            'total_processed': total_questions,
            'high_confidence_answers': high_confidence,
            'ambiguous_answers': ambiguous,
            'multiple_selections': multiple_selection,
            'unanswered': unanswered,
            'average_confidence': round(avg_confidence, 3),
            'processing_success_rate': round((high_confidence / total_questions * 100), 2) if total_questions > 0 else 0
        }
    
    def generate_detailed_report(self, extraction_results: Dict[str, Any], 
                               student_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generate a detailed report of the extraction results.
        
        Args:
            extraction_results: Results from extract_answers
            student_info: Optional student information
            
        Returns:
            Detailed report dictionary
        """
        if not extraction_results.get('success', False):
            return {
                'success': False,
                'error': extraction_results.get('error', 'Unknown error')
            }
        
        results = extraction_results['results']
        score_summary = extraction_results.get('score_summary')
        stats = extraction_results['processing_stats']
        
        # Subject-wise breakdown (assuming 20 questions per subject for 5 subjects)
        questions_per_subject = 20
        subjects = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4', 'Subject 5']
        subject_scores = []
        
        if score_summary:
            question_scores = score_summary['question_scores']
            for i, subject in enumerate(subjects):
                start_idx = i * questions_per_subject
                end_idx = min(start_idx + questions_per_subject, len(question_scores))
                
                subject_questions = question_scores[start_idx:end_idx]
                subject_correct = sum(1 for q in subject_questions if q['is_correct'])
                subject_attempted = sum(1 for q in subject_questions if q['attempted'])
                
                subject_scores.append({
                    'subject': subject,
                    'questions': len(subject_questions),
                    'correct': subject_correct,
                    'attempted': subject_attempted,
                    'score': f"{subject_correct}/{questions_per_subject}",
                    'percentage': round((subject_correct / questions_per_subject * 100), 2)
                })
        
        report = {
            'success': True,
            'timestamp': None,  # Would be set by calling code
            'student_info': student_info or {},
            'overall_score': score_summary,
            'subject_breakdown': subject_scores,
            'processing_statistics': stats,
            'quality_indicators': {
                'high_confidence_rate': stats.get('processing_success_rate', 0),
                'ambiguous_rate': round((stats.get('ambiguous_answers', 0) / stats.get('total_processed', 1) * 100), 2),
                'multiple_selection_rate': round((stats.get('multiple_selections', 0) / stats.get('total_processed', 1) * 100), 2),
                'unanswered_rate': round((stats.get('unanswered', 0) / stats.get('total_processed', 1) * 100), 2)
            },
            'recommendations': self._generate_recommendations(stats, score_summary)
        }
        
        return report
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                 score_summary: Dict[str, Any] = None) -> List[str]:
        """
        Generate recommendations based on processing statistics.
        
        Args:
            stats: Processing statistics
            score_summary: Score summary if available
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check processing quality
        if stats.get('processing_success_rate', 0) < 85:
            recommendations.append("Consider improving image quality - low confidence in bubble detection")
        
        if stats.get('ambiguous_answers', 0) > stats.get('total_processed', 0) * 0.1:
            recommendations.append("High number of ambiguous answers detected - check for multiple markings")
        
        if stats.get('multiple_selections', 0) > 0:
            recommendations.append(f"Found {stats['multiple_selections']} questions with multiple selections")
        
        if stats.get('unanswered', 0) > stats.get('total_processed', 0) * 0.2:
            recommendations.append("High number of unanswered questions - ensure all bubbles are properly marked")
        
        # Score-based recommendations
        if score_summary:
            if score_summary.get('accuracy_percentage', 0) < 50:
                recommendations.append("Low accuracy score - review study materials")
            elif score_summary.get('accuracy_percentage', 0) > 90:
                recommendations.append("Excellent performance!")
            
            if score_summary.get('attempt_rate_percentage', 0) < 90:
                recommendations.append("Consider attempting all questions to maximize score")
        
        if not recommendations:
            recommendations.append("Good quality processing with no issues detected")
        
        return recommendations
    
    def export_results_to_json(self, results: Dict[str, Any], filepath: str) -> bool:
        """
        Export results to JSON file.
        
        Args:
            results: Results dictionary
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            # Convert any non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting results: {str(e)}")
            return False
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (QuestionResult, AnswerChoice)):
            return obj.__dict__
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj