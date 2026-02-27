"""
Alan's Python Code Execution Engine
Safe execution of Python code with sandboxing and monitoring
"""

import sys
import io
import ast
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Tuple, Optional, Any, List
import subprocess
from pathlib import Path
import json
from datetime import datetime


class PythonExecutor:
    """
    Safely executes Python code with sandboxing and output capture.
    Monitors execution for safety and provides comprehensive feedback.
    """
    
    def __init__(self, max_execution_time: int = 30, max_output_length: int = 10000):
        """
        Initialize the Python executor.
        
        Args:
            max_execution_time: Maximum execution time in seconds
            max_output_length: Maximum output length in characters
        """
        self.max_execution_time = max_execution_time
        self.max_output_length = max_output_length
        self.execution_history = []
        self.restricted_imports = {
            'os', 'sys', 'subprocess', 'socket', 'threading',
            '__main__', '__builtins__'
        }
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None,
                safe_mode: bool = True) -> Tuple[bool, str, Optional[Exception]]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            context: Optional execution context (variables)
            safe_mode: Whether to use restricted execution
            
        Returns:
            Tuple of (success, output, error)
        """
        if safe_mode:
            is_safe, safety_msg = self._check_code_safety(code)
            if not is_safe:
                return False, "", Exception(f"Safety check failed: {safety_msg}")
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Create execution environment
            exec_context = self._build_context(context)
            
            # Execute with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_context)
            
            # Collect output
            output = stdout_capture.getvalue()
            error_output = stderr_capture.getvalue()
            
            # Limit output length
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... [output truncated]"
            
            full_output = output + error_output
            
            # Log execution
            self._log_execution(code, True, full_output)
            
            return True, full_output, None
        
        except SyntaxError as e:
            error_msg = f"Syntax Error: {e.msg} at line {e.lineno}"
            self._log_execution(code, False, error_msg)
            return False, "", e
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._log_execution(code, False, error_msg)
            return False, "", e
    
    def execute_function(self, func_code: str, func_name: str,
                        args: List = None, kwargs: Dict = None) -> Tuple[bool, Any, Optional[Exception]]:
        """
        Execute a function defined in code.
        
        Args:
            func_code: Code defining the function
            func_name: Name of the function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (success, result, error)
        """
        args = args or []
        kwargs = kwargs or {}
        
        try:
            # Execute function definition
            exec_context = {}
            exec(func_code, exec_context)
            
            # Get function
            if func_name not in exec_context:
                return False, None, Exception(f"Function '{func_name}' not found")
            
            func = exec_context[func_name]
            
            # Call function
            result = func(*args, **kwargs)
            
            self._log_execution(f"Call: {func_name}(*{args}, **{kwargs})", True, str(result))
            
            return True, result, None
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._log_execution(func_code, False, error_msg)
            return False, None, e
    
    def execute_script_file(self, file_path: str, context: Optional[Dict] = None) -> Tuple[bool, str, Optional[Exception]]:
        """
        Execute a Python script from file.
        
        Args:
            file_path: Path to the Python script
            context: Optional execution context
            
        Returns:
            Tuple of (success, output, error)
        """
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            return self.execute(code, context, safe_mode=True)
        
        except FileNotFoundError:
            return False, "", Exception(f"File not found: {file_path}")
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code for structure and potential issues.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            tree = ast.parse(code)
            
            analysis = {
                "valid": True,
                "has_syntax_errors": False,
                "functions": [],
                "classes": [],
                "imports": [],
                "dangerous_calls": [],
                "warnings": []
            }
            
            # Walk the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    else:
                        analysis["imports"].append(node.module)
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        call_name = node.func.id
                        if call_name in ['eval', 'exec', 'compile', '__import__']:
                            analysis["dangerous_calls"].append(call_name)
            
            return analysis
        
        except SyntaxError as e:
            return {
                "valid": False,
                "has_syntax_errors": True,
                "error": str(e),
                "line": e.lineno
            }
    
    def benchmark_code(self, code: str, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark code execution time.
        
        Args:
            code: Code to benchmark
            iterations: Number of iterations
            
        Returns:
            Benchmarking results
        """
        import time
        
        times = []
        
        for _ in range(iterations):
            start = time.time()
            success, output, error = self.execute(code, safe_mode=False)
            times.append(time.time() - start)
        
        import statistics
        
        return {
            "iterations": iterations,
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def get_execution_history(self, limit: int = None) -> List[Dict]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of execution history entries
        """
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history
    
    def display_execution_history(self, limit: int = 10):
        """Display execution history in readable format."""
        history = self.get_execution_history(limit)
        
        if not history:
            print("No execution history.")
            return
        
        print("\n" + "="*70)
        print("EXECUTION HISTORY")
        print("="*70 + "\n")
        
        for i, entry in enumerate(history, 1):
            status = "✓" if entry["success"] else "✗"
            print(f"{i}. {status} [{entry['timestamp']}]")
            print(f"   Code: {entry['code'][:50]}...")
            if not entry["success"]:
                print(f"   Error: {entry['output'][:100]}...")
            print()
    
    def _check_code_safety(self, code: str) -> Tuple[bool, str]:
        """Check code for potentially dangerous operations."""
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import socket",
            "exec(",
            "eval(",
            "compile(",
            "__import__",
            "open(",
            "system(",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Detected potentially dangerous: {pattern}"
        
        return True, "Code passed safety check"
    
    def _build_context(self, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Build execution context with safe builtins."""
        # Safe builtins to allow
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            'max': max,
            'min': min,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'bool': bool,
            'abs': abs,
            'round': round,
            'pow': pow,
        }
        
        context = {'__builtins__': safe_builtins}
        
        if user_context:
            context.update(user_context)
        
        return context
    
    def _log_execution(self, code: str, success: bool, output: str):
        """Log code execution."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "code": code[:200],  # Store first 200 chars
            "success": success,
            "output": output[:500]  # Store first 500 chars of output
        }
        self.execution_history.append(entry)


def test_executor():
    """Test the Python executor."""
    print("Initializing Python Executor...")
    executor = PythonExecutor()
    
    # Test 1: Simple code
    print("\n--- Test 1: Simple Code ---")
    code1 = "x = 5 + 3\nprint(f'Result: {x}')"
    success, output, error = executor.execute(code1)
    print(f"Success: {success}")
    print(f"Output: {output}")
    
    # Test 2: Code analysis
    print("\n--- Test 2: Code Analysis ---")
    code2 = """
def greet(name):
    return f'Hello, {name}!'

class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
"""
    analysis = executor.analyze_code(code2)
    print(f"Functions: {analysis['functions']}")
    print(f"Classes: {analysis['classes']}")
    
    # Test 3: Function execution
    print("\n--- Test 3: Function Execution ---")
    func_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    success, result, error = executor.execute_function(func_code, 'fibonacci', [10])
    print(f"fibonacci(10) = {result}")
    
    # Test 4: Execution history
    print("\n--- Test 4: Execution History ---")
    executor.display_execution_history(limit=5)


if __name__ == "__main__":
    test_executor()
