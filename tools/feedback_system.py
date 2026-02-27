"""
High-End Persistent Feedback Loop System
Continuously evaluates, records, and improves based on detailed feedback.
"""

import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics


@dataclass
class FeedbackEntry:
    """Represents a feedback entry."""
    timestamp: str
    task_id: str
    output: str
    quality_metrics: Dict[str, float]
    success_indicators: Dict[str, bool]
    suggestions: List[str]
    status: str  # "pending", "analyzed", "applied"
    impact_score: float


class FeedbackCollector:
    """
    Collects feedback from multiple sources and channels.
    Integrates user feedback, automatic evaluation, and outcome analysis.
    """
    
    def __init__(self, feedback_dir: str = "/workspaces/Alan/.feedback"):
        """
        Initialize feedback collector.
        
        Args:
            feedback_dir: Directory for feedback storage
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.feedback_log = self.feedback_dir / "feedback.jsonl"
        self.quality_assessments = self.feedback_dir / "quality.json"
        self.collected_feedback = []
        
        if not self.feedback_log.exists():
            self.feedback_log.touch()
        
        if not self.quality_assessments.exists():
            with open(self.quality_assessments, 'w') as f:
                json.dump({"assessments": []}, f)
    
    def collect_feedback(self, task_id: str, output: str,
                        quality_metrics: Dict[str, float],
                        feedback_source: str = "auto") -> str:
        """
        Collect feedback on output.
        
        Args:
            task_id: ID of the task
            output: Generated output
            quality_metrics: Dict of quality scores {metric: score}
            feedback_source: Source of feedback ("user", "auto", "test")
            
        Returns:
            Feedback ID
        """
        success_indicators = self._analyze_success(output, quality_metrics)
        suggestions = self._generate_suggestions(output, quality_metrics, success_indicators)
        
        # Calculate overall quality
        overall_quality = statistics.mean(quality_metrics.values()) if quality_metrics else 0.0
        
        feedback = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            output=output[:500],  # First 500 chars
            quality_metrics=quality_metrics,
            success_indicators=success_indicators,
            suggestions=suggestions,
            status="pending",
            impact_score=overall_quality
        )
        
        # Persist feedback
        with open(self.feedback_log, 'a') as f:
            f.write(json.dumps(asdict(feedback)) + '\n')
        
        self.collected_feedback.append(feedback)
        
        return feedback.timestamp[:10] + "_" + task_id
    
    def _analyze_success(self, output: str, quality_metrics: Dict) -> Dict[str, bool]:
        """Analyze success indicators."""
        return {
            "non_empty": len(output.strip()) > 0,
            "coherent": len(output.split()) > 5,
            "completed": "unable" not in output.lower() and "error" not in output.lower(),
            "high_quality": quality_metrics.get("relevance", 0) > 0.7,
            "meets_standards": all(v > 0.5 for v in quality_metrics.values())
        }
    
    def _generate_suggestions(self, output: str, quality_metrics: Dict,
                            success_indicators: Dict) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if not success_indicators["non_empty"]:
            suggestions.append("Generate more substantial content")
        
        if not success_indicators["coherent"]:
            suggestions.append("Improve coherence and structure")
        
        if quality_metrics.get("relevance", 1) < 0.7:
            suggestions.append("Improve relevance to query")
        
        if quality_metrics.get("completeness", 1) < 0.7:
            suggestions.append("Provide more complete response")
        
        if not success_indicators["high_quality"]:
            suggestions.append("Focus on quality over brevity")
        
        return suggestions


class FeedbackAnalyzer:
    """
    Analyzes feedback patterns and extracts actionable insights.
    Identifies trends and root causes of issues.
    """
    
    def __init__(self, feedback_collector: FeedbackCollector):
        """
        Initialize analyzer.
        
        Args:
            feedback_collector: Connected feedback collector
        """
        self.collector = feedback_collector
        self.analysis_cache = {}
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in collected feedback.
        
        Returns:
            Trends analysis
        """
        if not self.collector.collected_feedback:
            return {"status": "no_feedback_yet"}
        
        feedback_list = self.collector.collected_feedback
        
        # Quality trends
        quality_scores = [f.impact_score for f in feedback_list]
        avg_quality = statistics.mean(quality_scores)
        quality_trend = "improving" if quality_scores[-1] > avg_quality else "declining"
        
        # Success rate
        successful = sum(
            1 for f in feedback_list
            if all(f.success_indicators.values())
        )
        success_rate = successful / len(feedback_list)
        
        # Common suggestions
        all_suggestions = []
        for f in feedback_list:
            all_suggestions.extend(f.suggestions)
        
        suggestion_freq = {}
        for s in all_suggestions:
            suggestion_freq[s] = suggestion_freq.get(s, 0) + 1
        
        top_suggestions = sorted(
            suggestion_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        analysis = {
            "total_feedback_entries": len(feedback_list),
            "average_quality": avg_quality,
            "quality_trend": quality_trend,
            "success_rate": success_rate,
            "top_improvement_areas": [s[0] for s in top_suggestions],
            "suggestion_frequency": dict(top_suggestions),
            "quality_distribution": {
                "high": sum(1 for q in quality_scores if q > 0.8) / len(quality_scores),
                "medium": sum(1 for q in quality_scores if 0.5 <= q <= 0.8) / len(quality_scores),
                "low": sum(1 for q in quality_scores if q < 0.5) / len(quality_scores)
            }
        }
        
        return analysis
    
    def identify_failure_patterns(self) -> List[Dict]:
        """
        Identify patterns in failures.
        
        Returns:
            List of failure pattern analyses
        """
        patterns = []
        failed_feedback = [f for f in self.collector.collected_feedback
                          if not f.success_indicators["completed"]]
        
        if not failed_feedback:
            return []
        
        # Group by common issues
        issue_groups = {}
        for feedback in failed_feedback:
            for suggestion in feedback.suggestions:
                if suggestion not in issue_groups:
                    issue_groups[suggestion] = []
                issue_groups[suggestion].append(feedback)
        
        # Create patterns
        for issue, feedbacks in issue_groups.items():
            pattern = {
                "issue": issue,
                "frequency": len(feedbacks),
                "affected_tasks": [f.task_id for f in feedbacks],
                "avg_impact": statistics.mean(f.impact_score for f in feedbacks),
                "recommendation": self._recommend_fix(issue)
            }
            patterns.append(pattern)
        
        return sorted(patterns, key=lambda p: p["frequency"], reverse=True)
    
    def _recommend_fix(self, issue: str) -> str:
        """Recommend fix for identified issue."""
        fixes = {
            "Generate more substantial content": "Increase max_tokens or improve prompt specificity",
            "Improve coherence and structure": "Use better prompt structuring or increase reasoning steps",
            "Improve relevance to query": "Enhance context understanding or add query analysis step",
            "Provide more complete response": "Add completion verification step or extend generation",
            "Focus on quality over brevity": "Reduce speed optimization, increase reasoning depth"
        }
        return fixes.get(issue, "Review approach and consider alternative strategy")


class FeedbackLoopManager:
    """
    Manages the complete feedback loop.
    Coordinates collection, analysis, and application of feedback.
    """
    
    def __init__(self, learning_system=None, collector: FeedbackCollector = None,
                 analyzer: FeedbackAnalyzer = None):
        """
        Initialize feedback loop manager.
        
        Args:
            learning_system: Learning system to integrate with
            collector: Feedback collector
            analyzer: Feedback analyzer
        """
        self.learning_system = learning_system
        self.collector = collector or FeedbackCollector()
        self.analyzer = analyzer or FeedbackAnalyzer(self.collector)
        
        self.feedback_history = []
        self.applied_improvements = []
    
    def process_task_outcome(self, task_id: str, generated_output: str,
                            expected_output: Optional[str] = None,
                            user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Process complete task outcome with feedback.
        
        Args:
            task_id: Task ID
            generated_output: Generated output
            expected_output: Expected/reference output
            user_feedback: Optional user feedback
            
        Returns:
            Feedback processing result
        """
        # Evaluate output quality
        quality_metrics = self._evaluate_quality(
            generated_output,
            expected_output,
            user_feedback
        )
        
        # Collect feedback
        feedback_id = self.collector.collect_feedback(
            task_id=task_id,
            output=generated_output,
            quality_metrics=quality_metrics,
            feedback_source="composite"
        )
        
        # Analyze feedback
        trends = self.analyzer.analyze_feedback_trends()
        failure_patterns = self.analyzer.identify_failure_patterns()
        
        # Prepare improvement actions
        improvements = self._prepare_improvements(trends, failure_patterns)
        
        result = {
            "feedback_id": feedback_id,
            "quality_metrics": quality_metrics,
            "trends": trends,
            "failure_patterns": failure_patterns,
            "recommended_improvements": improvements,
            "status": "analyzed"
        }
        
        self.feedback_history.append(result)
        
        return result
    
    def _evaluate_quality(self, output: str, expected: Optional[str],
                         user_feedback: Optional[str]) -> Dict[str, float]:
        """Evaluate output quality from multiple dimensions."""
        metrics = {
            "length": min(len(output) / 500, 1.0),  # 500 chars is good
            "completeness": 0.8 if len(output) > 100 else 0.3,
            "coherence": self._measure_coherence(output),
            "relevance": 0.7,  # Would be improved with semantic similarity
        }
        
        # If expected output provided, measure similarity
        if expected:
            metrics["accuracy"] = self._measure_similarity(output, expected)
        
        # If user feedback provided, boost quality
        if user_feedback and "good" in user_feedback.lower():
            for key in metrics:
                metrics[key] = min(metrics[key] * 1.2, 1.0)
        
        return metrics
    
    def _measure_coherence(self, text: str) -> float:
        """Measure text coherence (0-1)."""
        # Simple heuristic: proper length and structure
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        avg_words_per_sentence = len(text.split()) / len(sentences)
        if 5 < avg_words_per_sentence < 25:
            return 0.9
        elif 3 < avg_words_per_sentence < 30:
            return 0.7
        else:
            return 0.4
    
    def _measure_similarity(self, output: str, expected: str) -> float:
        """Measure similarity between output and expected (0-1)."""
        # Simple word overlap metric
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.5
        
        overlap = len(output_words & expected_words)
        return min(overlap / len(expected_words), 1.0)
    
    def _prepare_improvements(self, trends: Dict, patterns: List) -> List[Dict]:
        """Prepare concrete improvements based on analysis."""
        improvements = []
        
        # From trends
        if trends.get("quality_trend") == "declining":
            improvements.append({
                "type": "adjust_strategy",
                "action": "Switch to more conservative approach",
                "priority": "high",
                "confidence": 0.85
            })
        
        if trends.get("success_rate", 1) < 0.7:
            improvements.append({
                "type": "increase_reasoning",
                "action": "Increase chain-of-thought steps",
                "priority": "high",
                "confidence": 0.80
            })
        
        # From failure patterns
        for pattern in patterns[:3]:  # Top 3 patterns
            improvements.append({
                "type": "fix_issue",
                "action": pattern["recommendation"],
                "issue": pattern["issue"],
                "affected_count": pattern["frequency"],
                "priority": "high" if pattern["frequency"] > 2 else "medium",
                "confidence": 0.7
            })
        
        return improvements
    
    def apply_improvements(self, improvements: List[Dict]) -> Dict:
        """
        Apply recommended improvements.
        
        Args:
            improvements: List of improvements to apply
            
        Returns:
            Application status
        """
        applied = []
        failed = []
        
        for imp in improvements:
            try:
                if imp["type"] == "adjust_strategy":
                    # Would integrate with agent
                    applied.append(imp)
                
                elif imp["type"] == "increase_reasoning":
                    # Would modify reasoning config
                    applied.append(imp)
                
                elif imp["type"] == "fix_issue":
                    # Record for learning system
                    if self.learning_system:
                        self.learning_system.record_event(
                            task_id="improvement_"+str(len(applied)),
                            goal="improve_on_issue",
                            approach=imp["action"],
                            outcome="applied",
                            success=True,
                            quality_score=0.75,
                            tokens_used=0,
                            feedback=imp["issue"],
                            lessons=[imp["action"]]
                        )
                    applied.append(imp)
            except Exception as e:
                failed.append((imp, str(e)))
        
        status = {
            "total_improvements": len(improvements),
            "applied": len(applied),
            "failed": len(failed),
            "applied_details": applied,
            "timestamp": datetime.now().isoformat()
        }
        
        self.applied_improvements.append(status)
        
        return status
    
    def get_feedback_report(self) -> Dict:
        """Get comprehensive feedback report."""
        return {
            "total_feedback_entries": len(self.collector.collected_feedback),
            "analysis": self.analyzer.analyze_feedback_trends(),
            "failure_patterns": self.analyzer.identify_failure_patterns(),
            "applied_improvements": len(self.applied_improvements),
            "improvement_history": self.applied_improvements[-5:]  # Last 5
        }
