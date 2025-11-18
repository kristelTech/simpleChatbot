# DeepSeek Chatbot

DeepSeek Chatbot is a local interactive chatbot application using the **DeepSeek-R1-Distill-Qwen-1.5B** model from Hugging Face. It allows you to chat with the model using either Apple Silicon’s MPS acceleration or CPU.

## Features

- Interactive chat with conversation history
- Local model loading with caching support
- Device-aware execution (MPS for Apple Silicon, CPU otherwise)
- Hugging Face authentication using API key
- Torch model compilation for faster inference

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/) with MPS support (for Apple Silicon) or CPU
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```
2. Put your HUGGINGFACE_API_KEY in .env

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the program
```bash
cd deepseek_chatbot
python chatbot.py
```

## How It Works

1. Loads Hugging Face API key from `.env`.
2. Chooses device: MPS if available, otherwise CPU.
3. Loads tokenizer and model from local cache.
4. Maintains conversation history and generates responses.
5. Uses `torch.inference_mode()` for efficient generation.

## Notes

- Ensure your model is cached locally to avoid repeated downloads.
- MPS acceleration is only available on Apple Silicon (M1/M2 Macs).
- Conversation history is stored in memory and resets when the script restarts.
- Adjust `max_new_tokens` in the script to control response length.