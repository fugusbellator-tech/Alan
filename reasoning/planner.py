"""
Sophisticated Goal Planning & Task Decomposition Engine
Breaks down complex objectives into executable subtasks with adaptive strategies.
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import torch


class GoalDecomposer:
    """
    Decomposes complex goals into hierarchical subtasks.
    Uses reasoning to understand dependencies and optimal execution order.
    """
    
    def __init__(self, model, tokenizer, max_depth: int = 5):
        """
        Initialize goal decomposer.
        
        Args:
            model: Language model for planning
            tokenizer: Tokenizer for the model
            max_depth: Maximum decomposition depth
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.decomposition_cache = {}
        
    def decompose_goal(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Decompose a complex goal into subtasks.
        
        Args:
            goal: The high-level goal/objective
            context: Additional context about the goal
            
        Returns:
            Dictionary with decomposed tasks, dependencies, and execution strategy
        """
        # Check cache first
        cache_key = hash(goal)
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        decomposition = {
            "goal": goal,
            "primary_tasks": [],
            "subtasks": {},
            "dependencies": {},
            "estimated_tokens": 0,
            "execution_order": [],
            "fallback_strategies": [],
            "success_criteria": []
        }
        
        # Analyze goal complexity
        complexity = self._analyze_goal_complexity(goal)
        
        if complexity < 0.3:
            # Simple goal - single task
            decomposition["primary_tasks"].append({
                "id": "task_1",
                "description": goal,
                "type": "direct",
                "estimated_tokens": 100,
                "required_tools": self._extract_required_tools(goal),
                "priority": 1
            })
        else:
            # Complex goal - decompose hierarchically
            decomposition = self._hierarchical_decomposition(goal, context)
        
        # Calculate dependencies
        decomposition["dependencies"] = self._calculate_dependencies(decomposition["primary_tasks"])
        decomposition["execution_order"] = self._determine_execution_order(
            decomposition["primary_tasks"],
            decomposition["dependencies"]
        )
        
        # Generate fallback strategies
        decomposition["fallback_strategies"] = self._generate_fallbacks(goal)
        
        # Define success criteria
        decomposition["success_criteria"] = self._define_success_metrics(goal)
        
        # Cache result
        self.decomposition_cache[cache_key] = decomposition
        
        return decomposition
    
    def _analyze_goal_complexity(self, goal: str) -> float:
        """
        Analyze goal complexity (0.0 to 1.0).
        
        Returns:
            Complexity score
        """
        complexity_indicators = {
            "dependencies": ["required", "after", "then", "given", "depends"],
            "multiple_steps": ["steps", "phases", "stages", "iterate", "multiple"],
            "conditional": ["if", "unless", "depends on", "conditional"],
            "abstract": ["improve", "optimize", "enhance", "better"]
        }
        
        score = 0.0
        goal_lower = goal.lower()
        
        for keyword_list in complexity_indicators.values():
            for keyword in keyword_list:
                if keyword in goal_lower:
                    score += 0.15
        
        return min(score, 1.0)
    
    def _hierarchical_decomposition(self, goal: str, context: Optional[Dict]) -> Dict:
        """Decompose goal hierarchically into subtasks."""
        decomposition = {
            "goal": goal,
            "primary_tasks": [],
            "subtasks": {},
            "dependencies": {},
            "estimated_tokens": 0,
            "execution_order": [],
            "fallback_strategies": [],
            "success_criteria": []
        }
        
        # Generate decomposition prompt
        decomp_prompt = f"""Goal: {goal}

Break this goal into 3-5 primary tasks. For each task, identify:
1. Task ID and name
2. Description
3. Dependencies (other tasks it depends on)
4. Subtasks needed
5. Estimated effort

Format as JSON:
{{
  "tasks": [
    {{"id": "task_1", "name": "...", "description": "...", "depends_on": [], "subtasks": [...], "effort": "low|medium|high"}}
  ]
}}"""
        
        try:
            inputs = self.tokenizer.encode(decomp_prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=300,
                    num_beams=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(decomp_prompt):]
            
            # Parse JSON from response
            task_list = self._extract_json_tasks(response)
            
            task_id = 1
            for task in task_list:
                task_obj = {
                    "id": f"task_{task_id}",
                    "name": task.get("name", f"Task {task_id}"),
                    "description": task.get("description", ""),
                    "type": "primary",
                    "depends_on": task.get("depends_on", []),
                    "subtasks": task.get("subtasks", []),
                    "estimated_tokens": self._estimate_tokens(task.get("effort", "medium")),
                    "required_tools": self._extract_required_tools(task.get("description", "")),
                    "priority": task_id
                }
                decomposition["primary_tasks"].append(task_obj)
                
                # Add subtasks
                for i, subtask in enumerate(task.get("subtasks", [])):
                    subtask_obj = {
                        "id": f"task_{task_id}_sub_{i+1}",
                        "description": subtask,
                        "parent_task": f"task_{task_id}",
                        "estimated_tokens": self._estimate_tokens("low")
                    }
                    decomposition["subtasks"][f"task_{task_id}_sub_{i+1}"] = subtask_obj
                
                task_id += 1
        
        except Exception as e:
            # Fallback: simple decomposition
            decomposition["primary_tasks"].append({
                "id": "task_1",
                "name": "Analyze",
                "description": f"Analyze: {goal}",
                "type": "primary",
                "estimated_tokens": 150,
                "required_tools": []
            })
            decomposition["primary_tasks"].append({
                "id": "task_2",
                "name": "Execute",
                "description": f"Execute solution for: {goal}",
                "type": "primary",
                "depends_on": ["task_1"],
                "estimated_tokens": 200,
                "required_tools": []
            })
        
        return decomposition
    
    def _extract_json_tasks(self, text: str) -> List[Dict]:
        """Extract task JSON from generated text."""
        try:
            # Try to find JSON block
            json_match = re.search(r'\{.*?"tasks".*?\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return data.get("tasks", [])
        except:
            pass
        return []
    
    def _extract_required_tools(self, description: str) -> List[str]:
        """Extract required tools from task description."""
        tools = []
        tool_keywords = {
            "code": ["execute", "python", "function", "calculate", "compute"],
            "network": ["fetch", "request", "download", "api", "http"],
            "meta_learning": ["improve", "optimize", "modify", "update", "learn"],
            "reasoning": ["analyze", "reason", "deduce", "infer", "think"],
            "scripting": ["script", "automate", "process", "batch"]
        }
        
        desc_lower = description.lower()
        for tool, keywords in tool_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    tools.append(tool)
                    break
        
        return list(set(tools))
    
    def _estimate_tokens(self, effort: str) -> int:
        """Estimate token consumption for effort level."""
        mapping = {
            "low": 100,
            "medium": 250,
            "high": 500,
            "very_high": 780
        }
        return mapping.get(effort, 200)
    
    def _calculate_dependencies(self, tasks: List[Dict]) -> Dict[str, List[str]]:
        """Calculate task dependencies."""
        dependencies = {}
        for task in tasks:
            task_id = task["id"]
            dependencies[task_id] = task.get("depends_on", [])
        return dependencies
    
    def _determine_execution_order(self, tasks: List[Dict], dependencies: Dict) -> List[str]:
        """Determine optimal execution order using topological sort."""
        order = []
        remaining = {t["id"]: t for t in tasks}
        
        while remaining:
            # Find tasks with no dependencies
            ready = [
                tid for tid, task in remaining.items()
                if not any(dep in remaining for dep in dependencies.get(tid, []))
            ]
            
            if not ready:
                # Circular dependency - just add remaining
                order.extend(remaining.keys())
                break
            
            # Add ready tasks ordered by priority
            ready.sort(key=lambda tid: remaining[tid].get("priority", float('inf')))
            order.extend(ready)
            
            for tid in ready:
                del remaining[tid]
        
        return order
    
    def _generate_fallbacks(self, goal: str) -> List[Dict]:
        """Generate fallback strategies for goal."""
        fallbacks = [
            {
                "strategy": "simplified_approach",
                "description": "Simplify the goal and try basic approach",
                "token_limit": 400,
                "priority": 1
            },
            {
                "strategy": "step_back",
                "description": "Break down further or try different angle",
                "token_limit": 200,
                "priority": 2
            },
            {
                "strategy": "adaptive_retry",
                "description": "Use feedback to adjust approach and retry",
                "token_limit": 300,
                "priority": 3
            }
        ]
        return fallbacks
    
    def _define_success_metrics(self, goal: str) -> List[Dict]:
        """Define success criteria for goal."""
        metrics = [
            {
                "name": "task_completion",
                "description": "All primary tasks completed",
                "weight": 0.5
            },
            {
                "name": "quality",
                "description": "Result quality meets expectations",
                "weight": 0.3
            },
            {
                "name": "efficiency",
                "description": "Resource usage within estimated limits",
                "weight": 0.2
            }
        ]
        return metrics


class ExecutionPlanner:
    """
    Plans execution strategy for decomposed tasks.
    Allocates resources and manages optimization.
    """
    
    def __init__(self, max_tokens_per_task: int = 780, reserved_tokens: int = 50):
        """
        Initialize execution planner.
        
        Args:
            max_tokens_per_task: Maximum tokens per task
            reserved_tokens: Tokens reserved for safety
        """
        self.max_tokens_per_task = max_tokens_per_task
        self.reserved_tokens = reserved_tokens
        self.execution_history = []
        
    def plan_execution(self, decomposition: Dict, context_tokens: int = 0) -> Dict:
        """
        Create execution plan with resource allocation.
        
        Args:
            decomposition: Decomposed goal
            context_tokens: Tokens used for context
            
        Returns:
            Execution plan with timing and resource allocation
        """
        available_tokens = self.max_tokens_per_task - context_tokens - self.reserved_tokens
        total_estimated = sum(t.get("estimated_tokens", 100) for t in decomposition["primary_tasks"])
        
        plan = {
            "goal": decomposition["goal"],
            "total_available_tokens": available_tokens,
            "total_estimated_tokens": total_estimated,
            "tasks": [],
            "phase": "execution_phase_1",
            "max_iterations": 3,
            "timeout_seconds": 300
        }
        
        # Calculate token allocation per task
        allocation_ratio = available_tokens / max(total_estimated, 1)
        
        for task in decomposition["primary_tasks"]:
            allocated_tokens = int(task.get("estimated_tokens", 100) * allocation_ratio)
            allocated_tokens = min(allocated_tokens, available_tokens // 2)
            
            task_plan = {
                "id": task["id"],
                "name": task.get("name", task["id"]),
                "description": task["description"],
                "allocated_tokens": allocated_tokens,
                "required_tools": task.get("required_tools", []),
                "retry_strategy": "adaptive",
                "max_retries": 2,
                "success_threshold": 0.7
            }
            plan["tasks"].append(task_plan)
        
        return plan
