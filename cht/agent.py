"""
Autonomous Agent Orchestrator
Coordinates sophisticated goal decomposition, execution, learning, and feedback loops.
Enables autonomous agent operation with iterative refinement.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from pathlib import Path
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reasoning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from planner import GoalDecomposer, ExecutionPlanner
from learning import AdaptiveLearningSystem, PerformanceOptimizer
from feedback_system import FeedbackCollector, FeedbackAnalyzer, FeedbackLoopManager


class AutonomousAgent:
    """
    Sophisticated autonomous agent that:
    - Decomposes complex goals into subtasks
    - Executes with adaptive strategies
    - Learns from feedback
    - Iterates to improve results
    - Persists knowledge across sessions
    """
    
    def __init__(self, model, tokenizer, chat_interface=None,
                 agent_dir: str = "/workspaces/Alan/.agent"):
        """
        Initialize autonomous agent.
        
        Args:
            model: Language model
            tokenizer: Model tokenizer
            chat_interface: Chat interface for generation
            agent_dir: Directory for agent data
        """
        self.model = model
        self.tokenizer = tokenizer
        self.chat_interface = chat_interface
        self.agent_dir = Path(agent_dir)
        self.agent_dir.mkdir(exist_ok=True)
        
        # Core components
        self.goal_decomposer = GoalDecomposer(model, tokenizer, max_depth=5)
        self.execution_planner = ExecutionPlanner()
        self.learning_system = AdaptiveLearningSystem()
        self.performance_optimizer = PerformanceOptimizer(self.learning_system)
        self.feedback_manager = FeedbackLoopManager(self.learning_system)
        
        # Agent state
        self.current_goal = None
        self.current_plan = None
        self.execution_history = []
        self.iteration_count = 0
        self.max_iterations = 3
        self.autonomy_enabled = False
        self.internet_enabled = False
        
        # Execution context
        self.available_tools = ["code_execution", "meta_learning", "reasoning"]
        
    def enable_autonomy(self, enabled: bool = True):
        """Enable/disable autonomous operation."""
        self.autonomy_enabled = enabled
        print(f"✓ Agent autonomy: {'ENABLED' if enabled else 'DISABLED'}")
    
    def enable_internet(self, enabled: bool = True):
        """Enable/disable internet access."""
        self.internet_enabled = enabled
        if enabled:
            self.available_tools.append("network")
        print(f"✓ Agent internet: {'ENABLED' if enabled else 'DISABLED'}")
    
    def pursue_goal(self, goal: str, context: Optional[Dict] = None,
                   callback: Optional[Callable] = None) -> Dict:
        """
        Autonomously pursue a goal.
        Decomposes, executes, evaluates, and iterates until success.
        
        Args:
            goal: The goal to pursue
            context: Additional context
            callback: Callback for progress updates
            
        Returns:
            Execution result
        """
        if not self.autonomy_enabled:
            return {"error": "Autonomy not enabled"}
        
        self.current_goal = goal
        self.iteration_count = 0
        
        print("\n" + "="*70)
        print(f"AUTONOMOUS GOAL PURSUIT: {goal}")
        print("="*70)
        
        # Step 1: Decompose goal
        print("\n[1/5] Decomposing goal into subtasks...")
        decomposition = self.goal_decomposer.decompose_goal(goal, context)
        self._print_decomposition(decomposition)
        
        # Step 2: Plan execution
        print("\n[2/5] Planning execution strategy...")
        plan = self.execution_planner.plan_execution(decomposition)
        self.current_plan = plan
        self._print_plan(plan)
        
        # Step 3: Get optimized strategy from learning
        print("\n[3/5] Checking learned strategies...")
        learned_strategy = self.performance_optimizer.optimize_for_goal(goal)
        print(f"    Strategy: {learned_strategy.get('action', 'default')}")
        if 'lessons_to_apply' in learned_strategy:
            print(f"    Lessons to apply: {len(learned_strategy['lessons_to_apply'])}")
        
        # Step 4: Execute with iteration
        print("\n[4/5] Executing tasks with adaptive iteration...")
        execution_result = self._execute_plan(plan, callback)
        
        # Step 5: Feedback & Learning
        print("\n[5/5] Processing feedback and learning...")
        feedback_result = self.feedback_manager.process_task_outcome(
            task_id=goal[:20],
            generated_output=execution_result.get("final_output", ""),
            user_feedback=None
        )
        
        self._print_feedback_analysis(feedback_result)
        
        # Apply improvements
        improvements = feedback_result.get("recommended_improvements", [])
        if improvements:
            print("\n    Applying improvements from feedback...")
            self.feedback_manager.apply_improvements(improvements)
        
        final_result = {
            "goal": goal,
            "success": execution_result.get("success", False),
            "output": execution_result.get("final_output", ""),
            "iterations": self.iteration_count,
            "quality_score": feedback_result.get("quality_metrics", {}).get("completeness", 0),
            "learning_applied": learned_strategy.get("action") != "generate_new_approach",
            "improvements_recommended": len(improvements),
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*70)
        print(f"GOAL COMPLETION: {'SUCCESS' if final_result['success'] else 'PARTIAL'}")
        print(f"Quality Score: {final_result['quality_score']:.2%}")
        print(f"Iterations: {self.iteration_count}/{self.max_iterations}")
        print("="*70 + "\n")
        
        return final_result
    
    def _execute_plan(self, plan: Dict, callback: Optional[Callable] = None) -> Dict:
        """
        Execute plan with adaptive iteration and error recovery.
        
        Args:
            plan: Execution plan
            callback: Progress callback
            
        Returns:
            Execution result
        """
        all_outputs = []
        failed_tasks = []
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            iteration_success = True
            
            print(f"\n    --- ITERATION {self.iteration_count}/{self.max_iterations} ---")
            
            for task_plan in plan["tasks"]:
                task_id = task_plan["id"]
                description = task_plan["description"]
                allocated_tokens = task_plan["allocated_tokens"]
                
                print(f"    Executing: {task_id}")
                
                # Execute task
                task_result = self._execute_task(
                    task_id=task_id,
                    description=description,
                    allocated_tokens=allocated_tokens,
                    tools=task_plan.get("required_tools", [])
                )
                
                # Record execution
                self.execution_history.append({
                    "iteration": self.iteration_count,
                    "task_id": task_id,
                    "success": task_result["success"],
                    "output": task_result["output"][:100],
                    "quality": task_result.get("quality", 0.5)
                })
                
                # Record learning event
                self.learning_system.record_event(
                    task_id=task_id,
                    goal=self.current_goal,
                    approach="decomposition_execution",
                    outcome=task_result["output"][:100],
                    success=task_result["success"],
                    quality_score=task_result.get("quality", 0.5),
                    tokens_used=allocated_tokens,
                    lessons=task_result.get("lessons", [])
                )
                
                all_outputs.append(task_result["output"])
                
                if not task_result["success"]:
                    iteration_success = False
                    failed_tasks.append(task_id)
                    
                    # Try retry with fallback
                    if task_plan.get("max_retries", 0) > 0:
                        print(f"      ↻ Retrying with fallback strategy...")
                        retry_result = self._retry_task_with_fallback(
                            task_id,
                            description,
                            task_plan.get("retry_strategy", "aggressive")
                        )
                        if retry_result["success"]:
                            all_outputs[-1] = retry_result["output"]
                            iteration_success = True
                            failed_tasks.remove(task_id)
            
            # Check if iteration was successful
            if iteration_success:
                print(f"    ✓ Iteration {self.iteration_count} successful!")
                break
            else:
                print(f"    ✗ Iteration {self.iteration_count} had failures")
                if iteration < self.max_iterations - 1:
                    print(f"      Retrying with adaptive strategy...")
        
        final_output = "\n".join(all_outputs)
        success = len(failed_tasks) == 0
        
        return {
            "success": success,
            "final_output": final_output,
            "all_outputs": all_outputs,
            "failed_tasks": failed_tasks,
            "iterations": self.iteration_count,
            "execution_history": self.execution_history
        }
    
    def _execute_task(self, task_id: str, description: str,
                     allocated_tokens: int, tools: List[str]) -> Dict:
        """
        Execute a single task.
        
        Args:
            task_id: Task ID
            description: Task description
            allocated_tokens: Tokens allocated
            tools: Available tools for this task
            
        Returns:
            Task execution result
        """
        try:
            # Generate response using chat interface or model directly
            if self.chat_interface:
                output = self.chat_interface.generate_response(
                    description,
                    use_reasoning=True
                )
            else:
                # Direct model generation
                inputs = self.tokenizer.encode(description, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=min(allocated_tokens, 200),
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Evaluate output quality
            quality = min(len(output) / 200, 1.0)  # Simple quality metric
            
            # Tools integration
            tool_results = []
            if "code_execution" in tools and any(kw in description.lower() 
                                                  for kw in ["execute", "calculate"]):
                # Would call executor here
                pass
            
            return {
                "success": True,
                "output": output,
                "quality": quality,
                "lessons": ["Task completed successfully"],
                "tools_used": [t for t in tools if t != "unknown"]
            }
        
        except Exception as e:
            return {
                "success": False,
                "output": f"Error executing task: {str(e)}",
                "quality": 0.1,
                "lessons": [f"Failed with error: {str(e)}"],
                "error": str(e)
            }
    
    def _retry_task_with_fallback(self, task_id: str, description: str,
                                  strategy: str) -> Dict:
        """
        Retry task with fallback strategy.
        
        Args:
            task_id: Task ID
            description: Task description
            strategy: Fallback strategy
            
        Returns:
            Retry result
        """
        fallback_prompts = {
            "aggressive": f"Simplify and execute: {description}",
            "conservative": f"Carefully approach: {description}",
            "creative": f"Try novel approach for: {description}"
        }
        
        fallback_prompt = fallback_prompts.get(strategy, description)
        
        # Execute with fallback
        return self._execute_task(
            task_id=task_id + "_retry",
            description=fallback_prompt,
            allocated_tokens=100,
            tools=[]
        )
    
    def _print_decomposition(self, decomp: Dict):
        """Print decomposition details."""
        print(f"\n  Goal: {decomp['goal']}")
        print(f"  Primary Tasks: {len(decomp['primary_tasks'])}")
        for task in decomp["primary_tasks"]:
            print(f"    - {task['name']}: {task['description'][:50]}...")
        print(f"  Execution Order: {' → '.join(decomp['execution_order'][:3])}...")
        print(f"  Success Criteria: {len(decomp['success_criteria'])} defined")
    
    def _print_plan(self, plan: Dict):
        """Print execution plan details."""
        print(f"\n  Available Tokens: {plan['total_available_tokens']}")
        print(f"  Tasks: {len(plan['tasks'])}")
        for task in plan["tasks"][:3]:
            print(f"    - {task['name']}: {task['allocated_tokens']} tokens")
        print(f"  Max Iterations: {plan['max_iterations']}")
    
    def _print_feedback_analysis(self, feedback: Dict):
        """Print feedback analysis."""
        print(f"\n  Quality Metrics: {feedback.get('quality_metrics', {})}")
        print(f"  Trends: {feedback.get('trends', {}).get('quality_trend', 'stable')}")
        print(f"  Improvement Areas: {feedback.get('trends', {}).get('top_improvement_areas', [])}")
    
    def get_agent_status(self) -> Dict:
        """Get current agent status."""
        return {
            "autonomy_enabled": self.autonomy_enabled,
            "internet_enabled": self.internet_enabled,
            "available_tools": self.available_tools,
            "current_goal": self.current_goal,
            "iteration_count": self.iteration_count,
            "execution_history_length": len(self.execution_history),
            "learning_status": self.learning_system.get_learning_summary(),
            "feedback_status": self.feedback_manager.get_feedback_report()
        }
    
    def display_status(self):
        """Display formatted agent status."""
        status = self.get_agent_status()
        
        print("\n" + "="*60)
        print("AUTONOMOUS AGENT STATUS")
        print("="*60)
        print(f"Autonomy: {'✓ ENABLED' if status['autonomy_enabled'] else '✗ DISABLED'}")
        print(f"Internet: {'✓ ENABLED' if status['internet_enabled'] else '✗ DISABLED'}")
        print(f"Available Tools: {', '.join(status['available_tools'])}")
        print(f"Current Goal: {status['current_goal'] or 'None'}")
        print(f"Iterations: {status['iteration_count']}")
        
        learning = status.get('learning_status', {})
        print(f"\nLearning Progress:")
        print(f"  Events Recorded: {learning.get('total_events_recorded', 0)}")
        print(f"  Success Rate: {learning.get('success_rate', 0):.1%}")
        print(f"  Patterns Learned: {learning.get('patterns_learned', 0)}")
        print(f"  Avg Quality: {learning.get('avg_quality', 0):.2f}")
        
        print("="*60 + "\n")
