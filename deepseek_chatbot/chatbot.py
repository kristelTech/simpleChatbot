import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from huggingface_hub import login
import threading

# ----------------------------------------
# Load environment variables from .env file
# ----------------------------------------
load_dotenv()

# Retrieve HuggingFace API key and log in
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
login(huggingface_api_key)

# ----------------------------------------
# Select compute device
# Prefer Apple MPS for speed on macOS;
# fall back to CPU if MPS is unavailable
# ----------------------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ----------------------------------------
# Set model name from HuggingFace
# (Tokenizer loads from HF; model loads locally)
# ----------------------------------------
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ----------------------------------------
# Load tokenizer
# trust_remote_code=True allows custom HF model code
# local_files_only=False lets it download on first run
# ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=False
)

# ----------------------------------------
# Load model from a local directory
# local_files_only=True ensures no HF download attempts
# ----------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    "../local_model",
    trust_remote_code=True,
    local_files_only=True
)

# Move model to device (MPS or CPU)
model.to(device)

# Optionally compile for speed (PyTorch 2.0+)
model = torch.compile(model)

# ----------------------------------------
# Store full conversation history in memory
# (We will use up to the last 20 exchanges)
# ----------------------------------------
conversation = []

print("Chat with the model! Type 'exit' to quit.\n")

# ----------------------------------------
# Main chat loop
# ----------------------------------------
while True:
    user_input = input("You: ")

    # Exit conditions
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to conversation history
    conversation.append({"role": "user", "content": user_input})

    # ----------------------------------------
    # Tokenize chat using the model's chat template
    # This ensures the model gets proper role formatting
    # ----------------------------------------
    inputs = tokenizer.apply_chat_template(
        conversation[-20:],          # Keep only last 20 messages
        add_generation_prompt=True,  # Insert assistant prompt
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # ----------------------------------------
    # Create a streamer so tokens are printed as they generate
    # ----------------------------------------
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # ----------------------------------------
    # Spawn generation in a separate thread
    # (Streamer consumes tokens asynchronously)
    # ----------------------------------------
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "max_new_tokens": 800,   # Longer limit to avoid truncation
            "streamer": streamer,
            "temperature": 0.7,      # Sampling randomness
            "top_p": 0.9,            # Nucleus sampling
            "do_sample": True
        }
    )
    thread.start()

    # ----------------------------------------
    # Print output tokens as they arrive
    # ----------------------------------------
    print("Model: ", end="", flush=True)
    generated_text = ""
    for token in streamer:
        print(token, end="", flush=True)
        generated_text += token
    print("\n")

    # Add assistant response to conversation history
    conversation.append({"role": "assistant", "content": generated_text})