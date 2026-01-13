# GitHub-Based VS Code + Colab Workflow Guide

This guide explains how to develop your keyboard suggestion ML models locally in VS Code while training them on Google Colab's free GPU resources using GitHub as a synchronization bridge.

## Table of Contents

1. [Overview](#overview)
2. [Initial Setup](#initial-setup)
3. [Development Workflow](#development-workflow)
4. [Training in Colab](#training-in-colab)
5. [Saving and Downloading Models](#saving-and-downloading-models)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### How It Works

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   VS Code   │ ──push→ │   GitHub    │ ←─open─ │Google Colab │
│   (Local)   │ ←─pull─ │(Repository) │ ──sync→ │   (Cloud)   │
└─────────────┘         └─────────────┘         └─────────────┘
     ↓                                                  ↓
  Edit Code                                      Train on GPU
  Git Commit                                    Download Models
```

### Workflow Summary

1. **Develop Locally**: Write code, create notebooks, organize files in VS Code
2. **Push to GitHub**: Commit and push changes to your repository
3. **Train in Colab**: Open notebooks from GitHub in Colab, run on GPU
4. **Download Models**: Save trained models and download to local machine
5. **Repeat**: Continue iterating with this cycle

### Advantages

✅ **Free GPU Access**: Use Colab's free T4 GPU for training  
✅ **Local Development**: Full VS Code IDE features (IntelliSense, Git, extensions)  
✅ **Version Control**: Everything tracked in Git  
✅ **No Manual Uploads**: GitHub handles synchronization  
✅ **Reproducible**: Notebooks contain complete training pipeline

---

## Initial Setup

### Step 1: Set Up GitHub Repository

#### Option A: Create New Repository on GitHub

1. Go to [github.com](https://github.com) and sign in
2. Click **"New repository"** (green button)
3. Configure repository:
   - **Name**: `KeyboardSuggestionsML` (or your preferred name)
   - **Visibility**: Choose **Private** (recommended) or Public
   - **Initialize**: Do NOT add README, .gitignore, or license (we'll create these)
4. Click **"Create repository"**

#### Option B: Use Existing Local Repository

If you already have the project locally:

```bash
cd /Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab

# Initialize git if not already done
git init

# Add remote (replace MinhPhuPham with your GitHub username)
git remote add origin https://github.com/MinhPhuPham/Keyboard-Suggestions-ML-Colab.git
```

### Step 2: Configure Local Repository

```bash
# Navigate to project directory
cd /Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab

# Check git status
git status

# Add all files (the .gitignore will exclude data/models)
git add .

# Commit
git commit -m "Initial commit: Project structure and workflow setup"

# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

### Step 3: Verify on GitHub

1. Go to your repository URL: `https://github.com/MinhPhuPham/Keyboard-Suggestions-ML-Colab`
2. Verify you see:
   - ✅ `docs/` directory with `PROJECT_PLAN.md` and `COLAB_WORKFLOW_GUIDE.md`
   - ✅ `notebooks/` directory with training notebooks
   - ✅ `src/` directory with helper scripts
   - ✅ `.gitignore`, `requirements.txt`, `.env.example`
   - ❌ NO `data/` or `models/` directories (they're ignored)

### Step 4: Connect Google Colab to GitHub

1. Open [Google Colab](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **"File" → "Open notebook"**
4. Click the **"GitHub"** tab
5. If prompted, click **"Authorize with GitHub"** and grant permissions
6. You should now see your repositories listed

---

## Development Workflow

### Daily Development Cycle

#### 1. Edit Code Locally (VS Code)

```bash
# Open project in VS Code
cd /Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab
code .

# Make changes to:
# - Helper scripts in src/
# - Notebooks in notebooks/
# - Documentation in docs/
```

#### 2. Test Locally (Optional)

```bash
# Create virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test helper scripts
python src/data_prep.py
```

#### 3. Commit and Push to GitHub

```bash
# Check what changed
git status

# Add specific files or all changes
git add src/data_prep.py
# OR
git add .

# Commit with descriptive message
git commit -m "Add emoji augmentation to data prep"

# Push to GitHub
git push
```

#### 4. Verify on GitHub

- Go to your repository on GitHub
- Confirm your latest commit appears
- Check that files are updated

---

## Training in Colab

### Opening Notebooks from GitHub

#### Method 1: Via Colab Interface

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **"File" → "Open notebook"**
3. Click **"GitHub"** tab
4. Search for your repository: `MinhPhuPham/Keyboard-Suggestions-ML-Colab`
5. Select the notebook: `notebooks/train_english.ipynb` or `notebooks/train_japanese.ipynb`
6. Click to open

#### Method 2: Direct URL

Use this URL pattern (replace placeholders):
```
https://colab.research.google.com/github/MinhPhuPham/Keyboard-Suggestions-ML-Colab/blob/main/notebooks/train_english.ipynb
```

### Running Training Notebooks

#### Step 1: Enable GPU Runtime

1. In Colab, click **"Runtime" → "Change runtime type"**
2. Set **"Hardware accelerator"** to **"GPU"** (T4 on free tier)
3. Click **"Save"**

#### Step 2: Run Setup Cells

The first cells in each notebook will:

```python
# 1. Clone your repository to Colab
!git clone https://github.com/MinhPhuPham/Keyboard-Suggestions-ML-Colab.git
%cd KeyboardSuggestionsML

# 2. Install dependencies
!pip install -q -r requirements.txt

# 3. Download UniDic (for Japanese only)
!python -m unidic download
```

**Run these cells first** (Shift+Enter or click play button)

#### Step 2.5: Kaggle API Setup (Optional for English Dataset)

The English training notebook downloads the SwiftKey corpus. The system has **automatic fallback**:
- **Primary**: Kaggle API (faster, if configured)
- **Fallback**: Direct download from Coursera CDN (automatic, no setup needed)

**Option 1: Skip Setup (Recommended for First Run)**

Just run the data setup cell - it will automatically download from Coursera CDN:

```python
# This will work without any Kaggle setup
data_path = setup_english_data(DRIVE_BASE)
```

You'll see:
```
⚠ Kaggle API not configured.
Option 2: Using direct download instead...
Downloading SwiftKey corpus from Coursera CDN...
✓ SwiftKey corpus downloaded
```

**Option 2: Set Up Kaggle API (For Faster Downloads)**

If you want faster downloads and access to more Kaggle datasets:

1. **Get API Token**:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll to "API" section
   - Click "Create New API Token"
   - **Copy the token** (Kaggle now uses environment variables instead of files)

2. **Set Environment Variable in Colab** (add this cell before data setup):

```python
# Set Kaggle API token
import os

# Replace 'your_token_here' with your actual token from Kaggle
os.environ['KAGGLE_API_TOKEN'] = 'your_token_here'

print("✓ Kaggle API configured!")

# Verify
!kaggle datasets list --page-size 1
```

3. **Re-run data setup** - it will now use Kaggle API

> [!NOTE]
> Kaggle recently changed from using `kaggle.json` files to environment variables. Just copy your token and set it as shown above!

> [!NOTE]
> The automatic fallback means you never need to set up Kaggle API unless you want faster downloads. The direct download works perfectly fine!

#### Step 3: Run Data Preparation

```python
# Download and prepare data
# For English: Downloads SwiftKey corpus
# For Japanese: Streams CC100 from Hugging Face

# This may take 5-10 minutes
```

#### Step 4: Run Training

```python
# Fine-tune model with LoRA
# English: ~30-60 minutes on GPU
# Japanese: ~1-2 hours on GPU

# Monitor progress in cell output
```

#### Step 5: Run Optimization & Export

```python
# Quantize and prune model
# Export to ONNX and Core ML
# Verify size and latency targets
```

### Monitoring Training

- **Loss**: Should decrease over epochs (target: perplexity < 20)
- **GPU Usage**: Check with `!nvidia-smi` in a cell
- **Time Remaining**: Colab shows estimated time for each cell
- **Session Timeout**: Free tier disconnects after 90 minutes of inactivity

> [!WARNING]
> **Colab Session Limits**: Free tier has usage limits. If you hit limits, wait 12-24 hours or consider Colab Pro ($9.99/month) for longer sessions and better GPUs.

---

## Saving and Downloading Models

### Option 1: Download Directly from Colab

After training completes, download the model to your local machine:

```python
# In Colab notebook (final cells)

# 1. Package model files
!zip -r english_model.zip models/english_model/

# 2. Download to your computer
from google.colab import files
files.download('english_model.zip')
```

Then on your local machine:

```bash
# Navigate to Downloads folder
cd ~/Downloads

# Move to project
mv english_model.zip /Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab/

# Extract
cd /Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab
unzip english_model.zip
```

### Option 2: Save to Google Drive (Recommended for Large Models)

```python
# In Colab notebook

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Save model to Drive
!cp -r models/english_model /content/drive/MyDrive/KeyboardML/

# 3. Download from Drive to local machine later
# Go to drive.google.com, navigate to folder, download
```

### Option 3: Push to GitHub (Small Models Only)

> [!CAUTION]
> Only use this for models < 100MB. GitHub has a 100MB file size limit.

```python
# In Colab notebook

# Configure git
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Add, commit, push
!git add models/english_model.onnx
!git commit -m "Add trained English model"
!git push
```

Then pull on local machine:

```bash
cd /Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab
git pull
```

---

## Best Practices

### Data Management

✅ **DO**:
- Download datasets directly in Colab notebooks (SwiftKey, CC100)
- Use Hugging Face `datasets` library with streaming for large data
- Keep data preparation code in notebooks for reproducibility

❌ **DON'T**:
- Commit large datasets to GitHub (use `.gitignore`)
- Manually upload data files to Colab (slow and error-prone)

### Code Organization

✅ **DO**:
- Keep reusable functions in `src/` helper scripts
- Import helpers in notebooks: `from src.data_prep import clean_english_text`
- Write modular, testable code

❌ **DON'T**:
- Duplicate code across notebooks
- Put all logic in notebook cells (hard to maintain)

### Version Control

✅ **DO**:
- Commit frequently with descriptive messages
- Use branches for experimental features
- Clear notebook outputs before committing (keeps diffs clean)

```bash
# Clear notebook outputs
jupyter nbconvert --clear-output --inplace notebooks/train_english.ipynb
git add notebooks/train_english.ipynb
git commit -m "Update training notebook"
```

❌ **DON'T**:
- Commit sensitive data (API keys, passwords)
- Push large binary files (models, datasets)

### Colab Sessions

✅ **DO**:
- Save checkpoints during long training runs
- Use `tqdm` progress bars to monitor training
- Test code locally before running expensive Colab training

❌ **DON'T**:
- Leave Colab idle (wastes GPU quota)
- Run multiple Colab sessions simultaneously on free tier

---

## Troubleshooting

### Issue: "Repository not found" in Colab

**Solution**:
1. Verify repository is public OR you've authorized Colab to access private repos
2. Go to [GitHub Settings → Applications](https://github.com/settings/applications)
3. Find "Google Colaboratory" and grant access to the repository

### Issue: "Git clone failed" in Colab

**Solution**:
```python
# For private repositories, use personal access token
!git clone https://YOUR_TOKEN@github.com/MinhPhuPham/Keyboard-Suggestions-ML-Colab.git

# Generate token at: https://github.com/settings/tokens
```

### Issue: "Module not found" when importing from src/

**Solution**:
```python
# Add src to Python path in Colab
import sys
sys.path.append('/content/Keyboard-Suggestions-ML-Colab/src')

# Now imports work
from data_prep import clean_english_text
```

### Issue: "CUDA out of memory" during training

**Solution**:
```python
# Reduce batch size in training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Reduce from 16 to 8
    gradient_accumulation_steps=2,   # Maintain effective batch size
)
```

### Issue: "Session disconnected" during long training

**Solution**:
1. **Save checkpoints**:
```python
training_args = TrainingArguments(
    save_steps=500,  # Save every 500 steps
    save_total_limit=2,  # Keep only 2 checkpoints
)
```

2. **Resume from checkpoint**:
```python
trainer.train(resume_from_checkpoint=True)
```

3. **Consider Colab Pro** for longer sessions

### Issue: "Cannot download files from Colab"

**Solution**:
1. Check file size (files > 100MB may fail)
2. Use Google Drive instead:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp model.zip /content/drive/MyDrive/
```

### Issue: Local changes conflict with Colab changes

**Solution**:
```bash
# Pull latest changes before starting work
git pull

# If conflicts occur
git stash          # Save local changes
git pull           # Get remote changes
git stash pop      # Reapply local changes
# Resolve conflicts manually
```

### Issue: Kaggle API errors or slow downloads

**Solution**:
The notebooks have automatic fallback to direct download. If you see Kaggle errors, just let it continue:

```
⚠ Kaggle API not configured.
Option 2: Using direct download instead...
Downloading SwiftKey corpus from Coursera CDN...
```

This is normal and will work fine!

**To set up Kaggle API** (optional):
1. Get token from [kaggle.com/settings](https://www.kaggle.com/settings)
2. Set environment variable in Colab:
```python
import os
os.environ['KAGGLE_API_TOKEN'] = 'your_token_here'  # Replace with your token
```

### Issue: Dataset download fails completely

**Solution**:
If both Kaggle and direct download fail, manually download and upload to Drive:

1. Download SwiftKey from [Kaggle](https://www.kaggle.com/datasets/therohk/tweets-blogs-news-swiftkey-dataset-4million)
2. Upload to Google Drive: `/Phu's Data development/data/english/`
3. Re-run notebook - it will detect existing data

---

## Quick Reference

### Common Git Commands

```bash
# Check status
git status

# Add files
git add .
git add src/data_prep.py

# Commit
git commit -m "Your message"

# Push
git push

# Pull latest changes
git pull

# View commit history
git log --oneline

# Create branch
git checkout -b feature-name

# Switch branch
git checkout main
```

### Common Colab Commands

```python
# Check GPU
!nvidia-smi

# Install package
!pip install package-name

# Clone repo
!git clone https://github.com/USER/REPO.git

# Change directory
%cd directory-name

# List files
!ls -lh

# Check disk space
!df -h

# Download file
from google.colab import files
files.download('filename.zip')

# Upload file
from google.colab import files
uploaded = files.upload()
```

---

## Next Steps

1. **Complete Initial Setup**: Follow [Initial Setup](#initial-setup) section
2. **Run First Training**: Try `train_english.ipynb` in Colab
3. **Iterate**: Make improvements based on results
4. **Deploy**: Export models to iOS/Android apps

For questions about the model architecture and training details, refer to [PROJECT_PLAN.md](file:///Users/phupham/side-projects/Keyboard-Suggestions-ML-Colab/docs/PROJECT_PLAN.md).
