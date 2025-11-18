import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
login(huggingface_api_key)

# Choose device: use MPS if available on Apple Silicon, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load tokenizer and model from local cache if possible
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=False  # set True after first run if cached
)

model = AutoModelForCausalLM.from_pretrained("../local_model", trust_remote_code=True, local_files_only=True)
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

    # Prepare inputs with conversation history
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # Generate response
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200)

    # Decode and print model response
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    print(f"Model: {generated_text}\n")

    # Add model response to conversation
    conversation.append({"role": "assistant", "content": generated_text})