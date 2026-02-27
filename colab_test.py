import os, sys, subprocess, shutil
from pathlib import Path
print("="*70)
print("ALAN AI SYSTEM - GOOGLE COLAB TEST")
print("="*70)

# ============================================================================
# STEP 1: Detect Environment
# ============================================================================
IS_COLAB = 'google.colab' in sys.modules
print(f"\n[1] Environment: {'Google Colab ☁️' if IS_COLAB else 'Local Machine 💻'}")

# ============================================================================
# STEP 2: Clone/Setup Repository
# ============================================================================
print("\n[2] Setting up Alan repository...")
if IS_COLAB:
    if os.path.exists('/content/Alan'):
        shutil.rmtree('/content/Alan')

    result = subprocess.run(
        'git clone https://github.com/fugusbellator-tech/Alan.git Alan',
        shell=True,
        capture_output=True,
        text=True,
        cwd='/content'
    )
    if result.returncode != 0:
        print(f"    ✗ Git clone failed with error:\n{result.stderr}")
        sys.exit(1)
    else:
        print(f"    ✓ Git clone successful.")

    os.chdir('/content/Alan')
    ALAN_PATH = '/content/Alan'
else:
    ALAN_PATH = os.getcwd()

sys.path.insert(0, ALAN_PATH)
print(f"    ✓ Working in: {ALAN_PATH}")
os.environ['ALAN_PATH'] = ALAN_PATH

# ============================================================================
# STEP 3: Install Dependencies
# ============================================================================
print("\n[3] Installing dependencies...")
packages = ['torch', 'transformers', 'numpy', 'safetensors', 'requests']
for pkg in packages:
    subprocess.run(f'pip install -q {pkg} 2>/dev/null', shell=True)
print("    ✓ All packages installed")

# ============================================================================
# STEP 4: Download Model
# ============================================================================
print("\n[4] Setting up language model...")
model_path = os.path.join(ALAN_PATH, 'models', 'gpt-neo-1.3b')

if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) and \
   not os.path.exists(os.path.join(model_path, 'model.safetensors')):
    print("    Downloading GPT-Neo 1.3B from HuggingFace...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3b")
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3b")
        model.save_pretrained(model_path)
        print("    ✓ Model downloaded and saved")
    except Exception as e:
        print(f"    ⚠ Download failed: {e}")
else:
    print("    ✓ Model already available")

# ============================================================================
# STEP 5: Import & Fix Alan Modules
# ============================================================================
print("\n[5] Importing and configuring Alan modules...")
try:
    # Ensure backup directories exist, including the one causing FileNotFoundError
    Path(ALAN_PATH).joinpath(".backups").mkdir(parents=True, exist_ok=True)
    # Explicitly create the '/workspaces/Alan/.backups' directory to prevent FileNotFoundError
    # if AlanAbilities incorrectly resolves its root_path.
    Path("/workspaces/Alan/.backups").mkdir(parents=True, exist_ok=True)

    # Import AlanAbilities first to patch it before AlanChat is imported
    from tools.abilities import AlanAbilities
    # Store the original __init__ method for proper patching
    _original_abilities_init = AlanAbilities.__init__

    # Define the patched __init__ that ensures ALAN_PATH is used
    def fixed_abilities_init(self, root_path=None):
        # Call the original __init__ with the corrected ALAN_PATH
        _original_abilities_init(self, root_path=ALAN_PATH)
        # Ensure self.root_path and self.backup_path are correctly set to ALAN_PATH
        self.root_path = Path(ALAN_PATH)
        self.backup_path = self.root_path / ".backups"
        self.backup_path.mkdir(exist_ok=True) # Ensure it's created based on ALAN_PATH

    # Apply the patch to AlanAbilities.__init__
    AlanAbilities.__init__ = fixed_abilities_init

    # Now import other modules, including AlanChat, which will use the patched AlanAbilities
    from tools.python_executor import PythonExecutor
    from reasoning.chain_of_thought import ChainOfThought
    from cht.chat import AlanChat

    print("    ✓ All modules configured with correct paths")
except Exception as e:
    print(f"    ✗ Setup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 6: Initialize Alan Chat
# ============================================================================
print("\n[6] Initializing Alan Chat...")
try:
    # Ensure we use the base class or specific init if it accepts paths
    chat = AlanChat(model_path=model_path, use_cot=False, use_gpu=False)
    print("    ✓ Alan Chat ready!")
except Exception as e:
    print(f"    ✗ Error during initialization: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 7: Interactive Chat Session
# ============================================================================
if 'chat' in locals():
    print("\n[7] Starting interactive chat session...")
    print("\n" + "="*70)
    print("ALAN INTERACTIVE CHAT")
    print("="*70)
    print("Type 'exit' to quit, 'help' for commands, 'clear' to clear history\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'exit':
                print("\nAlan: Goodbye! It was nice chatting with you.")
                break
            
            elif user_input.lower() == 'clear':
                chat.chat_history = []
                chat.conversation_context = ""
                print("Chat history cleared.")
                continue
            
            elif user_input.lower() == 'reasoning':
                chat.use_cot = not chat.use_cot
                status = "enabled" if chat.use_cot else "disabled"
                print(f"Chain of Thought reasoning {status}.")
                continue
            
            elif user_input.lower() == 'history':
                if chat.chat_history:
                    print("\n" + "="*60)
                    print("CHAT HISTORY")
                    print("="*60)
                    for i, entry in enumerate(chat.chat_history, 1):
                        print(f"{i}. You: {entry['user']}")
                        print(f"   Alan: {entry['response']}\n")
                else:
                    print("No chat history yet.")
                continue
            
            elif user_input.lower() == 'help':
                print("\n" + "="*60)
                print("AVAILABLE COMMANDS")
                print("="*60)
                print("  exit       - Exit the chat")
                print("  clear      - Clear chat history")
                print("  reasoning  - Toggle Chain of Thought reasoning")
                print("  history    - View chat history")
                print("  help       - Show this help message")
                print("="*60 + "\n")
                continue
            
            # Generate response
            response = chat.generate_response(user_input)
            print(f"\n🤖 Alan: {response}\n")
            
            # Add to history
            chat.chat_history.append({
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "user": user_input,
                "response": response,
                "cot_used": chat.use_cot
            })
        
        except KeyboardInterrupt:
            print("\n\nAlan: Goodbye!")
            break

print("\n" + "="*70)
print("✨ ALAN SESSION ENDED ✨")
print("="*70)