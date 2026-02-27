"""
Alan's Meta-Learning Abilities
Allows Alan to modify its own source code and weights for self-improvement
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tempfile


class AlanAbilities:
    """
    Meta-learning capabilities for Alan to modify its own source code and weights.
    Enables self-improvement and adaptation.
    """
    
    def __init__(self, root_path: str = "/workspaces/Alan"):
        """
        Initialize Alan's abilities.
        
        Args:
            root_path: Root directory of Alan project
        """
        self.root_path = Path(root_path)
        self.tools_path = self.root_path / "tools"
        self.reasoning_path = self.root_path / "reasoning"
        self.cht_path = self.root_path / "cht"
        self.models_path = self.root_path / "models"
        
        # Backup directory for safety
        self.backup_path = self.root_path / ".backups"
        self.backup_path.mkdir(exist_ok=True)
        
        # Modifications log
        self.modifications_log = self.backup_path / "modifications.json"
        self._initialize_log()

        # Autonomy/network state - NOW ENABLED FOR AGENT MODE
        self.autonomy_enabled = True
        self.internet_enabled = True
    
    def _initialize_log(self):
        """Initialize the modifications log."""
        if not self.modifications_log.exists():
            with open(self.modifications_log, 'w') as f:
                json.dump({"modifications": []}, f, indent=2)
    
    def modify_reasoning_logic(self, modification_type: str, code_snippet: str, 
                              description: str) -> bool:
        """
        Modify the chain of thought reasoning logic.
        
        Args:
            modification_type: "append", "replace", "inject", or "optimize"
            code_snippet: The new code to add/modify
            description: Description of the modification
            
        Returns:
            Success status
        """
        try:
            cot_file = self.reasoning_path / "chain_of_thought.py"
            
            # Create backup
            self._backup_file(cot_file, "chain_of_thought_reasoning")
            
            with open(cot_file, 'r') as f:
                content = f.read()
            
            if modification_type == "append":
                # Append to class
                new_content = content + f"\n    # Auto-added: {description}\n{code_snippet}\n"
            
            elif modification_type == "replace":
                # Find and replace function/method
                new_content = self._smart_replace(content, code_snippet, description)
            
            elif modification_type == "inject":
                # Inject into existing method
                new_content = self._inject_into_method(content, code_snippet, description)
            
            elif modification_type == "optimize":
                # Replace with optimized version
                new_content = self._optimize_code(content, code_snippet, description)
            
            else:
                return False
            
            # Write modified code
            with open(cot_file, 'w') as f:
                f.write(new_content)
            
            # Log modification
            self._log_modification("reasoning", modification_type, description, code_snippet)
            
            print(f"✓ Reasoning logic modified: {description}")
            return True
        
        except Exception as e:
            print(f"✗ Error modifying reasoning logic: {e}")
            return False
    
    def modify_chat_system(self, modification_type: str, code_snippet: str,
                          description: str) -> bool:
        """
        Modify the chat system logic.
        
        Args:
            modification_type: "append", "replace", "inject", or "optimize"
            code_snippet: The new code to add/modify
            description: Description of the modification
            
        Returns:
            Success status
        """
        try:
            chat_file = self.cht_path / "chat.py"
            
            # Create backup
            self._backup_file(chat_file, "chat_system")
            
            with open(chat_file, 'r') as f:
                content = f.read()
            
            if modification_type == "append":
                new_content = content + f"\n    # Auto-added: {description}\n{code_snippet}\n"
            
            elif modification_type == "replace":
                new_content = self._smart_replace(content, code_snippet, description)
            
            elif modification_type == "inject":
                new_content = self._inject_into_method(content, code_snippet, description)
            
            elif modification_type == "optimize":
                new_content = self._optimize_code(content, code_snippet, description)
            
            else:
                return False
            
            with open(chat_file, 'w') as f:
                f.write(new_content)
            
            self._log_modification("chat", modification_type, description, code_snippet)
            
            print(f"✓ Chat system modified: {description}")
            return True
        
        except Exception as e:
            print(f"✗ Error modifying chat system: {e}")
            return False
    
    def update_model_weights(self, weights_path: str, adapter: Optional[str] = None) -> bool:
        """
        Update or replace model weights.
        
        Args:
            weights_path: Path to new weights
            adapter: Optional LoRA adapter or fine-tuning parameters
            
        Returns:
            Success status
        """
        try:
            source = Path(weights_path)
            if not source.exists():
                print(f"✗ Weights file not found: {weights_path}")
                return False
            
            # Backup current weights
            self._backup_file(self.models_path / "gpt-neo-1.3b", "model_weights")

            # Copy new weights
            target = self.models_path / "gpt-neo-1.3b"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            
            # If adapter provided, save it
            if adapter:
                adapter_path = self.models_path / "adapter.json"
                with open(adapter_path, 'w') as f:
                    json.dump({"adapter": adapter}, f)
            
            self._log_modification("weights", "replace", 
                                 f"Model weights updated from {weights_path}", 
                                 f"File size: {source.stat().st_size}")
            
            print(f"✓ Model weights updated successfully")
            return True
        
        except Exception as e:
            print(f"✗ Error updating model weights: {e}")
            return False
    
    def fine_tune_weights(self, dataset_path: str, epochs: int = 1, 
                         learning_rate: float = 1e-5) -> bool:
        """
        Fine-tune model weights on custom dataset.
        
        Args:
            dataset_path: Path to training dataset
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Success status
        """
        try:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
            from transformers import Trainer, TrainingArguments
            import torch
            
            print(f"Starting fine-tuning: {epochs} epochs, LR={learning_rate}")
            
            # Load model and tokenizer
            model_path = str(self.models_path / "gpt-neo-1.3b")
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
            
            # Prepare dataset
            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=dataset_path,
                block_size=128
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Setup training
            training_args = TrainingArguments(
                output_dir=str(self.backup_path / "fine_tune_checkpoints"),
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                learning_rate=learning_rate,
                save_steps=100,
                save_total_limit=2,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            # Train
            trainer.train()
            
            # Save fine-tuned model
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            self._log_modification("weights", "fine_tune",
                                 f"Model fine-tuned on {dataset_path}",
                                 f"Epochs: {epochs}, LR: {learning_rate}")
            
            print(f"✓ Model fine-tuning completed")
            return True
        
        except ImportError:
            print("✗ Training libraries not available")
            return False
        except Exception as e:
            print(f"✗ Error during fine-tuning: {e}")
            return False
    
    def add_new_capability(self, capability_name: str, code: str,
                          integration_point: str = "chat") -> bool:
        """
        Add a new capability to Alan.
        
        Args:
            capability_name: Name of the new capability
            code: Python code implementing the capability
            integration_point: Where to integrate ("chat", "reasoning", or "tools")
            
        Returns:
            Success status
        """
        try:
            if integration_point == "tools":
                # Create new capability file in tools
                capability_file = self.tools_path / f"{capability_name}.py"
                with open(capability_file, 'w') as f:
                    f.write(code)
                print(f"✓ New capability added: {capability_name}")
                return True
            
            elif integration_point == "chat":
                return self.modify_chat_system("append", code, 
                                              f"New capability: {capability_name}")
            
            elif integration_point == "reasoning":
                return self.modify_reasoning_logic("append", code,
                                                  f"New capability: {capability_name}")
            
            else:
                return False
        
        except Exception as e:
            print(f"✗ Error adding capability: {e}")
            return False
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """
        Restore from a previous backup.
        
        Args:
            backup_name: Name of the backup to restore
            
        Returns:
            Success status
        """
        try:
            backup_file = self.backup_path / f"{backup_name}.backup"
            if not backup_file.exists():
                print(f"✗ Backup not found: {backup_name}")
                return False
            
            # Extract and restore
            print(f"Restoring from backup: {backup_name}")
            # Implementation depends on backup format
            return True
        
        except Exception as e:
            print(f"✗ Error restoring backup: {e}")
            return False
    
    def get_modification_history(self) -> List[Dict]:
        """Get history of all modifications."""
        try:
            with open(self.modifications_log, 'r') as f:
                data = json.load(f)
            return data.get("modifications", [])
        except:
            return []
    
    def display_modification_history(self):
        """Display modification history in readable format."""
        history = self.get_modification_history()
        
        if not history:
            print("No modifications recorded.")
            return
        
        print("\n" + "="*70)
        print("ALAN MODIFICATION HISTORY")
        print("="*70 + "\n")
        
        for i, mod in enumerate(history[-10:], 1):  # Show last 10
            print(f"{i}. [{mod['timestamp']}]")
            print(f"   Type: {mod['type']}")
            print(f"   Module: {mod['module']}")
            print(f"   Description: {mod['description']}")
            print()
    
    def _backup_file(self, file_path: Path, backup_name: str):
        """Create a backup of a file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_path / f"{backup_name}_{timestamp}.backup"
            
            if file_path.is_dir():
                shutil.copytree(file_path, backup_file)
            else:
                shutil.copy2(file_path, backup_file)
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    def _smart_replace(self, content: str, code_snippet: str, description: str) -> str:
        """Intelligently replace code sections."""
        # This is a placeholder - actual implementation would parse and replace
        return content
    
    def _inject_into_method(self, content: str, code_snippet: str, 
                           description: str) -> str:
        """Inject code into an existing method."""
        # Placeholder for method injection
        return content
    
    def _optimize_code(self, content: str, code_snippet: str, 
                      description: str) -> str:
        """Replace with optimized version."""
        # Placeholder for code optimization
        return content
    
    def _log_modification(self, module: str, mod_type: str, description: str,
                         code: str):
        """Log a modification."""
        try:
            with open(self.modifications_log, 'r') as f:
                data = json.load(f)
            
            modification = {
                "timestamp": datetime.now().isoformat(),
                "module": module,
                "type": mod_type,
                "description": description,
                "code_hash": hashlib.md5(code.encode()).hexdigest()
            }
            
            data["modifications"].append(modification)
            
            with open(self.modifications_log, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Could not log modification: {e}")

    # ------------------------------------------------------------------
    # Autonomy helpers
    # ------------------------------------------------------------------
    def enable_autonomy(self):
        """Allow Alan to take actions on its own behalf."""
        self.autonomy_enabled = True
        print("✓ Autonomy enabled")

    def disable_autonomy(self):
        """Turn off autonomous behaviour."""
        self.autonomy_enabled = False
        print("✓ Autonomy disabled")

    def grant_internet_access(self) -> bool:
        """
        Configure a basic internet-capable toolkit for Alan.  This will try to
        install the `requests` library if missing, create a simple network
        helper under `tools/network.py`, and mark the flag so other parts of the
        system can make use of it.
        """
        try:
            # make sure requests is available
            try:
                import requests  # type: ignore
            except ImportError:
                import subprocess, sys
                print("Installing requests library...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
                import requests  # type: ignore

            network_code = (
                "import requests\n"
                "\n"
                "def fetch(url, params=None, headers=None):\n"
                "    resp = requests.get(url, params=params, headers=headers)\n"
                "    return resp.text\n"
                "\n"
                "def post(url, data=None, json=None, headers=None):\n"
                "    resp = requests.post(url, data=data, json=json, headers=headers)\n"
                "    return resp.text\n"
            )
            net_file = self.tools_path / "network.py"
            with open(net_file, 'w') as f:
                f.write(network_code)

            self.internet_enabled = True
            self._log_modification("tools", "append", "internet toolkit added", network_code)
            print("✓ Internet access capability added (network.py created)")
            return True
        except Exception as e:
            print(f"✗ Could not grant internet access: {e}")
            return False

    def execute_shell_command(self, command: str) -> str:
        """
        Run an arbitrary shell command and return its output.  This is a dangerous
        primitive but useful for giving Alan the ability to inspect or modify the
        environment.
        """
        import subprocess, shlex
        try:
            proc = subprocess.run(shlex.split(command), capture_output=True, text=True)
            output = proc.stdout.strip() or proc.stderr.strip()
            return output
        except Exception as e:
            return f"Error executing command: {e}"

    def plan_actions(self, instruction: str) -> list:
        """
        Use Alan's reasoning capability to translate a natural-language
        instruction into a structured list of actions.  Each action is a dict
        with an "action" key and optional parameters.

        This method spins up a temporary AlanChat with reasoning enabled but
        with tools/autonomy disabled to avoid recursion.  The prompt requests a
        JSON array enclosed in <actions> tags so it can be reliably parsed.
        """
        actions = []
        try:
            # make sure the project root is on sys.path so imports work when
            # abilities is run as a standalone script
            import sys
            sys.path.insert(0, str(self.root_path))

            # lazy import to avoid heavy dependencies if unused
            from cht.chat import AlanChat

            # ensure correct model path (absolute) to avoid HF hub errors
            chat = AlanChat(model_path=str(self.models_path / "gpt-neo-1.3b"), use_cot=True)
            # disable autonomous execution during planning
            chat.tools_enabled = False
            chat.abilities = None
            chat.network = None

            plan_prompt = f"""Instruction: {instruction}

You are Alan, an AI with the ability to perform the following actions on yourself:
  - grant_internet_access()
  - execute_shell_command(command:str)
  - fetch_url(url:str)
  - modify_reasoning(mod_type:str, code:str, description:str)
  - add_capability(capability_name:str, code:str, integration_point:str)
  - enable_autonomy(), disable_autonomy()

Your goal is to interpret the instruction and, if appropriate, return an action plan.  **You MUST append a JSON array of actions as the last thing in your response.** It may be provided in one of two formats:
  1. Enclosed in `<actions>...</actions>` tags (preferred)
  2. On a line beginning with `ACTION:` followed by the JSON
If both appear, the `<actions>` version will be used.  Examples:
<actions>[{{"action":"grant_internet_access"}}]</actions>
ACTION: {{"action":"grant_internet_access"}}
If no action is needed, end with `<actions>[]</actions>` or `ACTION: []`."""
            plan_text = chat.generate_response(plan_prompt)

            # extract JSON between all <actions> tags (choose last if multiple)
            import re, json
            matches = re.findall(r"<actions>(.*?)</actions>", plan_text, re.DOTALL)
            if matches:
                raw = matches[-1].strip()
                try:
                    actions = json.loads(raw)
                except Exception:
                    print("Warning: could not parse action plan:", raw)
        except Exception as e:
            print(f"plan_actions failed: {e}")
        return actions

    def interpret_and_execute(self, instruction: str) -> bool:
        """
        Interpret a natural language instruction by asking Alan to plan
        a sequence of actions, then execute those actions.
        Returns True if at least one action was performed.
        """
        if not self.autonomy_enabled:
            return False

        actions = self.plan_actions(instruction)
        executed_any = False

        if actions:
            executed_any = self.execute_action_plan(actions)
        else:
            # fallback to previous simple heuristics if planning yielded nothing
            text = instruction.lower()
            if "internet" in text or "network" in text:
                executed_any |= self.grant_internet_access()
            if "fetch" in text and self.internet_enabled:
                parts = instruction.split()
                for i, w in enumerate(parts):
                    if w.lower() == 'fetch' and i + 1 < len(parts):
                        url = parts[i + 1]
                        try:
                            from tools import network
                            print(f"[network] fetching {url}")
                            result = network.fetch(url)
                            print(f"[network] response length {len(result)}")
                            executed_any = True
                        except Exception as e:
                            print(f"[network] fetch failed: {e}")
            if "shell" in text or "command" in text:
                parts = instruction.split(':', 1)
                if len(parts) > 1:
                    result = self.execute_shell_command(parts[1].strip())
                    print(f"Shell result: {result}")
                    executed_any = True
            if "modify reasoning" in text or "cot" in text:
                snippet = "# autonomous modification requested by interpret_and_execute"
                executed_any |= self.modify_reasoning_logic("append", snippet, "autonomous cot edit")

        return executed_any

    def execute_action_plan(self, actions: list) -> bool:
        """
        Execute a pre-parsed list of actions (i.e. no further planning).
        Each element should be a dict containing an "action" key and
        optional "args" list.
        """
        executed_any = False
        for act in actions:
            name = act.get("action")
            args = act.get("args", [])

            if name == "grant_internet_access":
                executed_any |= self.grant_internet_access()
            elif name == "execute_shell_command":
                if args:
                    result = self.execute_shell_command(args[0])
                    print(f"Shell result: {result}")
                    executed_any = True
            elif name == "fetch_url":
                if args and self.internet_enabled:
                    try:
                        from tools import network
                        print(f"[network] fetching {args[0]}")
                        result = network.fetch(args[0])
                        print(f"[network] response length {len(result)}")
                        executed_any = True
                    except Exception as e:
                        print(f"[network] fetch failed: {e}")
            elif name == "modify_reasoning":
                executed_any |= self.modify_reasoning_logic(
                    act.get("mod_type", ""),
                    act.get("code", ""),
                    act.get("description", "")
                )
            elif name == "add_capability":
                executed_any |= self.add_new_capability(
                    act.get("capability_name", ""),
                    act.get("code", ""),
                    act.get("integration_point", "chat")
                )
            elif name == "enable_autonomy":
                self.enable_autonomy()
                executed_any = True
            elif name == "disable_autonomy":
                self.disable_autonomy()
                executed_any = True
            else:
                print(f"Unknown action: {name}")
        return executed_any

    def self_improve(self, prompt: str) -> bool:
        """
        Use the existing chat interface to ask Alan what it would do given a
        request, then try to perform whatever concrete actions are suggested.
        Requires that the `cht` package is importable; the dependency is loaded
        lazily so abilities can still be used standalone.
        """
        try:
            from cht.chat import AlanChat
            chat = AlanChat(model_path=str(self.models_path / "gpt-neo-1.3b"), use_cot=False)
            response = chat.generate_response(prompt)
            print(f"[self_improve] alan suggested: {response}")
            return self.interpret_and_execute(response)
        except Exception as e:
            print(f"✗ self_improve failed: {e}")
            return False



def test_abilities():
    """Test the abilities module and new autonomy features."""
    print("Initializing Alan's Abilities...")
    abilities = AlanAbilities()
    
    print("\nTesting modification logging...")
    abilities._log_modification(
        "test", "append", 
        "Test modification", 
        "print('Hello from Alan')"
    )
    
    print("\nModification History:")
    abilities.display_modification_history()

    print("\nEnabling autonomy and granting internet...")
    abilities.enable_autonomy()
    abilities.grant_internet_access()

    print("\nAttempting to run a shell command...")
    print(abilities.execute_shell_command("echo autonomy test"))

    print("\nInterpreting a compound instruction...")
    abilities.interpret_and_execute("Please give internet access and run shell: echo hi")

    print("\nUsing self_improve to act on a prompt...")
    abilities.self_improve("Give yourself internet access capabilities and integrate them")

    print("\nTesting the new model-driven planner...")
    plan = abilities.plan_actions("Please fetch https://example.com and then run shell: echo done")
    print("Plan returned:", plan)
    print("Executing plan...")
    abilities.interpret_and_execute("Please fetch https://example.com and then run shell: echo done")


if __name__ == "__main__":
    test_abilities()
