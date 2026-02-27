"""
Comprehensive Adaptive Learning System
Learns from experience, feedback, and task outcomes to improve performance.
"""

import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class LearningEvent:
    """Represents a learning event from task execution."""
    timestamp: str
    task_id: str
    goal: str
    approach: str
    outcome: str
    success: bool
    quality_score: float
    tokens_used: int
    feedback: str
    lessons: List[str]
    improvements: List[str]


class AdaptiveLearningSystem:
    """
    Comprehensive system for adaptive learning.
    Records, analyzes, and applies lessons from experience.
    """
    
    def __init__(self, learning_db_path: str = "/workspaces/Alan/.learning"):
        """
        Initialize adaptive learning system.
        
        Args:
            learning_db_path: Path to learning database
        """
        self.learning_db = Path(learning_db_path)
        self.learning_db.mkdir(exist_ok=True)
        
        # Learning databases
        self.events_db = self.learning_db / "events.jsonl"
        self.patterns_db = self.learning_db / "patterns.json"
        self.strategies_db = self.learning_db / "strategies.json"
        self.metrics_db = self.learning_db / "metrics.json"
        
        self._initialize_dbs()
        
        # In-memory cache
        self.learned_patterns = self._load_patterns()
        self.learned_strategies = self._load_strategies()
        self.performance_metrics = self._load_metrics()
        self.recent_events = []
        
    def _initialize_dbs(self):
        """Initialize database files."""
        if not self.events_db.exists():
            self.events_db.touch()
        
        if not self.patterns_db.exists():
            with open(self.patterns_db, 'w') as f:
                json.dump({"patterns": {}}, f)
        
        if not self.strategies_db.exists():
            with open(self.strategies_db, 'w') as f:
                json.dump({"strategies": {}}, f)
        
        if not self.metrics_db.exists():
            with open(self.metrics_db, 'w') as f:
                json.dump({"metrics": {}}, f)
    
    def record_event(self, task_id: str, goal: str, approach: str, outcome: str,
                    success: bool, quality_score: float, tokens_used: int,
                    feedback: str = "", lessons: List[str] = None) -> str:
        """
        Record a learning event from task execution.
        
        Args:
            task_id: ID of the executed task
            goal: Goal being pursued
            approach: Approach/strategy used
            outcome: Actual outcome
            success: Whether task succeeded
            quality_score: Quality (0-1)
            tokens_used: Tokens consumed
            feedback: Optional feedback
            lessons: Lessons learned
            
        Returns:
            Event ID
        """
        lessons = lessons or []
        improvements = self._identify_improvements(goal, approach, success, quality_score)
        
        event = LearningEvent(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            goal=goal,
            approach=approach,
            outcome=outcome,
            success=success,
            quality_score=quality_score,
            tokens_used=tokens_used,
            feedback=feedback,
            lessons=lessons,
            improvements=improvements
        )
        
        # Log event
        with open(self.events_db, 'a') as f:
            f.write(json.dumps(asdict(event)) + '\n')
        
        self.recent_events.append(event)
        
        # Update patterns
        self._update_patterns(event)
        
        # Update strategy metrics
        self._update_strategy_performance(approach, success, quality_score)
        
        return hashlib.md5(event.timestamp.encode()).hexdigest()[:8]
    
    def _identify_improvements(self, goal: str, approach: str,
                             success: bool, quality_score: float) -> List[str]:
        """Identify possible improvements for next attempt."""
        improvements = []
        
        if not success:
            improvements.append("Try different decomposition strategy")
            improvements.append("Allocate more tokens to analysis")
            improvements.append("Use fallback approach")
        
        if quality_score < 0.5:
            improvements.append("Improve output quality focus")
            improvements.append("Add verification step")
        
        if quality_score < 0.7 and success:
            improvements.append("Refine approach for better quality")
            improvements.append("Add quality checking mechanism")
        
        return improvements
    
    def _update_patterns(self, event: LearningEvent):
        """Update learned patterns based on event."""
        # Create pattern key
        goal_hash = hashlib.md5(event.goal.encode()).hexdigest()[:6]
        pattern_key = f"{goal_hash}_{event.approach}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "goal_pattern": event.goal,
                "approach": event.approach,
                "success_count": 0,
                "attempt_count": 0,
                "avg_quality": 0.0,
                "last_used": None,
                "lessons": [],
                "effectiveness_score": 0.0
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["attempt_count"] += 1
        if event.success:
            pattern["success_count"] += 1
        
        # Update running average quality
        old_avg = pattern["avg_quality"]
        pattern["avg_quality"] = (
            (old_avg * (pattern["attempt_count"] - 1) + event.quality_score) /
            pattern["attempt_count"]
        )
        
        # Update effectiveness
        success_rate = pattern["success_count"] / pattern["attempt_count"]
        pattern["effectiveness_score"] = success_rate * pattern["avg_quality"]
        
        # Add unique lessons
        for lesson in event.lessons:
            if lesson not in pattern["lessons"]:
                pattern["lessons"].append(lesson)
        
        pattern["last_used"] = event.timestamp
        
        # Persist patterns
        self._save_patterns()
    
    def _update_strategy_performance(self, approach: str, success: bool,
                                    quality_score: float):
        """Update strategy performance metrics."""
        if approach not in self.learned_strategies:
            self.learned_strategies[approach] = {
                "name": approach,
                "uses": 0,
                "successes": 0,
                "failures": 0,
                "avg_quality": 0.0,
                "success_rate": 0.0,
                "rank": 0.0
            }
        
        strat = self.learned_strategies[approach]
        strat["uses"] += 1
        
        if success:
            strat["successes"] += 1
        else:
            strat["failures"] += 1
        
        # Update quality average
        new_avg = (strat["avg_quality"] * (strat["uses"] - 1) + quality_score) / strat["uses"]
        strat["avg_quality"] = new_avg
        
        strat["success_rate"] = strat["successes"] / strat["uses"]
        strat["rank"] = strat["success_rate"] * strat["avg_quality"]
        
        self._save_strategies()
    
    def get_best_strategy_for_goal(self, goal: str) -> Optional[Dict]:
        """
        Get best learned strategy for a goal.
        
        Args:
            goal: The goal/task
            
        Returns:
            Best strategy dict or None
        """
        # Find patterns matching goal
        matching_patterns = []
        for pattern_key, pattern in self.learned_patterns.items():
            if self._goals_similar(goal, pattern["goal_pattern"]):
                matching_patterns.append(pattern)
        
        if not matching_patterns:
            return None
        
        # Return highest effectiveness
        best = max(matching_patterns, key=lambda p: p["effectiveness_score"])
        return {
            "approach": best["approach"],
            "effectiveness_score": best["effectiveness_score"],
            "success_rate": best["success_count"] / best["attempt_count"],
            "avg_quality": best["avg_quality"],
            "lessons": best["lessons"]
        }
    
    def get_all_lessons(self) -> Dict[str, List[str]]:
        """Get all lessons learned across patterns."""
        lessons_by_approach = {}
        for pattern in self.learned_patterns.values():
            approach = pattern["approach"]
            if approach not in lessons_by_approach:
                lessons_by_approach[approach] = []
            lessons_by_approach[approach].extend(pattern["lessons"])
        return lessons_by_approach
    
    def _goals_similar(self, goal1: str, goal2: str) -> bool:
        """Check if two goals are similar."""
        # Simple substring matching for demo
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())
        overlap = len(words1 & words2)
        return overlap >= 2
    
    def _save_patterns(self):
        """Save patterns to disk."""
        with open(self.patterns_db, 'w') as f:
            json.dump({"patterns": self.learned_patterns}, f, indent=2)
    
    def _load_patterns(self) -> Dict:
        """Load patterns from disk."""
        try:
            with open(self.patterns_db, 'r') as f:
                return json.load(f).get("patterns", {})
        except:
            return {}
    
    def _save_strategies(self):
        """Save strategies to disk."""
        with open(self.strategies_db, 'w') as f:
            json.dump({"strategies": self.learned_strategies}, f, indent=2)
    
    def _load_strategies(self) -> Dict:
        """Load strategies from disk."""
        try:
            with open(self.strategies_db, 'r') as f:
                return json.load(f).get("strategies", {})
        except:
            return {}
    
    def _save_metrics(self):
        """Save metrics to disk."""
        with open(self.metrics_db, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
    
    def _load_metrics(self) -> Dict:
        """Load metrics from disk."""
        try:
            with open(self.metrics_db, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress."""
        total_events = len(self.recent_events)
        successful_events = sum(1 for e in self.recent_events if e.success)
        
        return {
            "total_events_recorded": total_events,
            "successful_outcomes": successful_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "patterns_learned": len(self.learned_patterns),
            "strategies_evaluated": len(self.learned_strategies),
            "best_strategy": max(
                self.learned_strategies.values(),
                key=lambda s: s["rank"],
                default=None
            ),
            "avg_quality": np.mean([e.quality_score for e in self.recent_events]) if self.recent_events else 0
        }


class PerformanceOptimizer:
    """
    Optimizes performance based on learned metrics.
    Adjusts strategies and allocations dynamically.
    """
    
    def __init__(self, learning_system: AdaptiveLearningSystem):
        """
        Initialize optimizer.
        
        Args:
            learning_system: Connected learning system
        """
        self.learning_system = learning_system
        self.optimization_history = []
    
    def optimize_for_goal(self, goal: str, current_approach: str = None) -> Dict:
        """
        Optimize strategy for goal using learned experience.
        
        Args:
            goal: Goal to optimize for
            current_approach: Current approach to improve
            
        Returns:
            Optimized strategy dict
        """
        best_strategy = self.learning_system.get_best_strategy_for_goal(goal)
        
        if best_strategy:
            optimization = {
                "action": "use_learned_strategy",
                "recommended_approach": best_strategy["approach"],
                "effectiveness": best_strategy["effectiveness_score"],
                "lessons_to_apply": best_strategy["lessons"]
            }
        else:
            # Generate new improved approach
            optimization = {
                "action": "generate_new_approach",
                "reasoning": "No prior successful approach found, attempting novel strategy",
                "apply_lessons": self.learning_system.get_all_lessons()
            }
        
        self.optimization_history.append({
            "goal": goal,
            "optimization": optimization,
            "timestamp": datetime.now().isoformat()
        })
        
        return optimization
