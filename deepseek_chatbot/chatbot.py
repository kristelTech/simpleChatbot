import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from huggingface_hub import login
import threading

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
login(huggingface_api_key)

# Choose device: use MPS if available on Apple Silicon, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=False  # set True after first run if cached
)

model = AutoModelForCausalLM.from_pretrained(
    "../local_model",  # path to your local model
    trust_remote_code=True,
    local_files_only=True
)
model.to(device)
model = torch.compile(model)

# Conversation history
conversation = []

print("Chat with the model! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to conversation
    conversation.append({"role": "user", "content": user_input})

    # Prepare inputs with conversation history (last 20 messages to avoid context overflow)
    inputs = tokenizer.apply_chat_template(
        conversation[-20:],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # Create streamer for real-time token output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate response in a separate thread
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "max_new_tokens": 800,  # increase token limit to avoid cut-off
            "streamer": streamer,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    )
    thread.start()

    # Print generated tokens as they come
    print("Model: ", end="", flush=True)
    generated_text = ""
    for token in streamer:
        print(token, end="", flush=True)
        generated_text += token
    print("\n")

    # Add model response to conversation
    conversation.append({"role": "assistant", "content": generated_text})