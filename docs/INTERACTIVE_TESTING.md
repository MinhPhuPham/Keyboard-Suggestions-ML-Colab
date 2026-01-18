# Interactive Model Testing

This script allows you to test your trained keyboard suggestion model interactively in the terminal.

## Installation

First, make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the interactive tester with the default model:

```bash
python test_model_interactive.py
```

### Specify a Different Model

```bash
python test_model_interactive.py --model-dir ./models/your_model_name
```

### Adjust Number of Suggestions

```bash
python test_model_interactive.py --num-suggestions 5
```

## Interactive Commands

Once the script is running, you can use these commands:

| Command | Description |
|---------|-------------|
| `<text>` | Type any text and press Enter to get suggestions |
| `:q` or `quit` | Quit the program |
| `:help` | Show help message |
| `:stats` | Show model statistics |

## Example Session

```
🎹 Keyboard Suggestion Model - Interactive Tester
============================================================
Instructions:
  - Type your text and press Enter to get suggestions
  - Type ':q' or 'quit' to exit
  - Type ':help' for more options
============================================================

📝 Input: Hello how are

🔮 Predictions:
  1. "Hello how are you doing today"
  2. "Hello how are you feeling"
  3. "Hello how are things going"

⏱️  Inference time: 45.23 ms
============================================================

📝 Input: I am going to

🔮 Predictions:
  1. "I am going to the store"
  2. "I am going to be there"
  3. "I am going to work tomorrow"

⏱️  Inference time: 42.18 ms
============================================================

📝 Input: :q

👋 Goodbye!
```

## Features

✅ **Real-time predictions** - Get instant suggestions as you type  
✅ **Timing information** - See how fast the model responds (in milliseconds)  
✅ **Multiple suggestions** - Get 3 different completion options  
✅ **Clean output** - Easy-to-read formatted predictions  
✅ **Interactive interface** - Simple commands for easy testing  

## Troubleshooting

### Model not found

If you get a "Model directory not found" error, make sure:
1. You have trained a model (run the training notebook first)
2. The model is saved in `./models/english_model/` directory
3. The directory contains both `english_model.pt` and `tokenizer/` folder

### Dependencies missing

If you get import errors, install dependencies:

```bash
pip install torch transformers
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Model Requirements

The script expects the following structure:

```
models/
└── english_model/
    ├── english_model.pt      # Trained model weights
    └── tokenizer/            # Tokenizer files
        ├── tokenizer_config.json
        ├── vocab.json
        └── merges.txt
```

## Performance Notes

- **Inference time**: Typically 20-100ms on CPU
- **GPU support**: The script uses CPU by default (modify for GPU if needed)
- **Memory usage**: ~500MB for DistilGPT2 model

## Customization

You can modify the prediction parameters in the script:

```python
# In the predict() method
max_new_tokens=10,      # How many tokens to generate
num_return_sequences=3, # Number of suggestions
temperature=0.8,        # Randomness (0.0-1.0)
top_k=50,              # Top-k sampling
top_p=0.95,            # Nucleus sampling
```

## Next Steps

After testing your model:
1. Note the inference times
2. Evaluate the quality of suggestions
3. If needed, retrain with different hyperparameters
4. Export to mobile format (ONNX/CoreML) for deployment
