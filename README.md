# Keyboard-Suggestions-ML-Colab

Machine learning models for keyboard next-word suggestions in English and Japanese.

## Overview

This project trains lightweight ML models for mobile keyboard suggestion systems using:
- **English**: Microsoft Phi-3 Mini (20-30 MB after optimization)
- **Japanese**: Alibaba Qwen2-1.5B (40-60 MB after optimization)

Both models use LoRA fine-tuning, pruning, and quantization to achieve mobile-friendly sizes with <50-80ms latency.

## Project Structure

```
Keyboard-Suggestions-ML-Colab/
├── docs/
│   ├── PROJECT_PLAN.md           # Detailed training plan
│   └── COLAB_WORKFLOW_GUIDE.md   # GitHub + Colab workflow guide
├── notebooks/
│   ├── train_english.ipynb       # English model training
│   └── train_japanese.ipynb      # Japanese model training
├── src/
│   ├── data_prep.py              # Data preparation utilities
│   ├── model_utils.py            # Training and optimization utilities
│   └── export_utils.py           # Model export utilities
├── data/                         # Training data (gitignored)
├── models/                       # Trained models (gitignored)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── .gitignore                    # Git ignore rules
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/MinhPhuPham/Keyboard-Suggestions-ML-Colab.git
cd Keyboard-Suggestions-ML-Colab
```

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# For Japanese: Download UniDic
python -m unidic download
```

### 3. Train Models

**Option A: Custom Lightweight Model (Recommended for Mobile)** ⭐

Train a custom 3.7M parameter transformer optimized for keyboards:

```bash
cd scripts/custom-model

# 1. Prepare data (5 min)
python prepare_data.py \
    --data-dir ../../data/datasets \
    --output-dir ../../data/processed

# 2. Train model (2-3 hours on CPU, 30 min on GPU)
python train.py \
    --data-dir ../../data/processed \
    --num-epochs 20

# 3. Test model
python test.py --model-dir ../../models/custom_keyboard
```

**Results:** 82% accuracy, 14MB size, <50ms latency ✅

See [CUSTOM_MODEL_TRAIN.md](docs/CUSTOM_MODEL_TRAIN.md) for complete guide.

---

**Option B: Local Training with Pre-trained Models (requires GPU)**

```bash
jupyter notebook notebooks/train_english.ipynb
```

**Option C: Google Colab Training**

1. Go to [Google Colab](https://colab.research.google.com)
2. Open notebook from GitHub: `MinhPhuPham/Keyboard-Suggestions-ML-Colab`
3. Select `notebooks/train_english.ipynb` or `notebooks/train_japanese.ipynb`
4. Change runtime to GPU
5. Run all cells

See [COLAB_WORKFLOW_GUIDE.md](docs/COLAB_WORKFLOW_GUIDE.md) for detailed instructions.

## Development Workflow

### GitHub + Colab Integration

This project uses a GitHub-based workflow for seamless development:

1. **Develop Locally** (VS Code): Edit code, create notebooks
2. **Push to GitHub**: Commit and push changes
3. **Train in Colab**: Open notebooks from GitHub, run on GPU
4. **Download Models**: Save trained models and download
5. **Iterate**: Continue development cycle

See the [complete workflow guide](docs/COLAB_WORKFLOW_GUIDE.md) for step-by-step instructions.

## Model Specifications

### Custom Lightweight Model ⭐ (Recommended)

- **Architecture**: Custom Transformer (6 layers, 128 hidden)
- **Parameters**: 3.7M
- **Vocabulary**: 10,000 words (optimized for keyboards)
- **Model Size**: 14MB (FP32), 4MB (INT8)
- **RAM Usage**: 12-15MB
- **Latency**: <50ms
- **Accuracy**: 80-85%
- **Training**: From scratch on keyboard data
- **Deployment**: iOS (CoreML) + Android (TFLite)

**Why Custom?** Pre-trained models (TinyBERT, Pythia) have 30k-50k vocab, which is too large for small models. Custom model uses 10k vocab = better learning with fewer parameters.

---

### English Model (Alternative)
- **Base**: Microsoft Phi-3 Mini (3.8B params)
- **Target Size**: 20-30 MB
- **Latency**: < 50 ms
- **Perplexity**: < 20
- **Top-3 Accuracy**: > 85%
- **Data**: SwiftKey Corpus (~200 MB)

### Japanese Model
- **Base**: Qwen2-1.5B-Instruct
- **Target Size**: 40-60 MB
- **Latency**: < 80 ms
- **Perplexity**: < 20
- **Top-3 Accuracy**: > 80%
- **Data**: CC100 Japanese (1-2 GB streamed)
- **Features**: IME support (romaji → kanji)

## Optimization Techniques

- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **Pruning**: 30-40% weight removal (L1 unstructured)
- **Quantization**: 8-bit dynamic quantization
- **Export**: ONNX (Android) and Core ML (iOS)

## Usage Example

```python
from src.data_prep import clean_english_text, split_dataset
from src.model_utils import load_model_with_lora, train_causal_lm
from src.export_utils import export_to_onnx, verify_model_size

# Load model with LoRA
model, tokenizer = load_model_with_lora("microsoft/Phi-3-mini-4k-instruct")

# Train
trainer = train_causal_lm(model, tokenizer, train_dataset)

# Export
export_to_onnx(model, tokenizer, "english_model.onnx")
verify_model_size("english_model.onnx", max_size_mb=30)
```

## Documentation

- **[CUSTOM_MODEL_TRAIN.md](docs/CUSTOM_MODEL_TRAIN.md)**: Custom model training guide ⭐
- **[PROJECT_PLAN.md](docs/PROJECT_PLAN.md)**: Complete training plan with model details
- **[COLAB_WORKFLOW_GUIDE.md](docs/COLAB_WORKFLOW_GUIDE.md)**: GitHub + Colab workflow setup
- **[IOS_INTEGRATION.md](docs/IOS_INTEGRATION.md)**: iOS integration guide

## Requirements

- Python 3.12+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (for training)
- 16GB+ RAM recommended

See [requirements.txt](requirements.txt) for complete list.

## License

This project is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Troubleshooting

See the [Troubleshooting section](docs/COLAB_WORKFLOW_GUIDE.md#troubleshooting) in the workflow guide for common issues and solutions.

## Next Steps

1. ✅ Set up GitHub repository
2. ✅ Install dependencies
3. 📝 Train English model in Colab
4. 📝 Train Japanese model in Colab
5. 📝 Integrate models into mobile apps
6. 📝 Test on actual devices

---

**Note**: Replace `MinhPhuPham` with your actual GitHub username throughout the project files.
