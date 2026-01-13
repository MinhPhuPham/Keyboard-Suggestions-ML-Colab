# Testing Guide

## Local Testing (Before Colab)

Before running the notebooks in Google Colab, you can test the workflow locally to catch errors early.

### Quick Test (No Dependencies)

Test file structure only:

```bash
python test_local.py
```

Expected output:
```
✓ PASS: File Structure
```

The other tests will fail if you don't have PyTorch installed locally (which is fine - we'll use Colab's GPU).

### Full Local Test (Optional)

If you want to test everything locally:

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m unidic download  # For Japanese
   ```

3. **Run tests**:
   ```bash
   python test_local.py
   ```

Expected output:
```
✓ PASS: Imports
✓ PASS: Data Preparation
✓ PASS: Dataset Creation
✓ PASS: Tokenizer
✓ PASS: File Structure

🎉 All tests passed! Ready for Colab training.
```

## Colab Testing

### Test English Notebook

1. Open [train_english.ipynb](https://colab.research.google.com/github/MinhPhuPham/Keyboard-Suggestions-ML-Colab/blob/main/notebooks/train_english.ipynb) in Colab

2. **Set GPU runtime**:
   - Runtime → Change runtime type → GPU → Save

3. **Run first 6 cells** (up to data setup):
   - Cell 1: Check environment ✓
   - Cell 2: Clone repository ✓
   - Cell 3: Install dependencies ✓
   - Cell 4: Import libraries ✓
   - Cell 5: Mount Drive ✓
   - Cell 6: Setup data ✓

4. **Verify**:
   - No errors in imports
   - Drive mounted successfully
   - Data path shown (existing or downloaded)

5. **If successful**, continue with training cells

### Test Japanese Notebook

Same process with [train_japanese.ipynb](https://colab.research.google.com/github/MinhPhuPham/Keyboard-Suggestions-ML-Colab/blob/main/notebooks/train_japanese.ipynb)

Additional cell:
- Cell 4: Download UniDic ✓

## Common Issues

### Issue: Module import errors

**Symptom**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: 
- In Colab: Run cell 3 (install dependencies) and wait for completion
- Locally: Install requirements: `pip install -r requirements.txt`

### Issue: Drive mount fails

**Symptom**: `Failed to mount Google Drive`

**Solution**:
1. Clear browser cookies
2. Try incognito mode
3. Manually click folder icon in Colab → Mount Drive

### Issue: Kaggle API not configured

**Symptom**: `Kaggle API not configured`

**Solution**:
```python
# In Colab, upload kaggle.json
from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Issue: CUDA out of memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in training cell
- Change from 8 to 4 or 2
- Restart runtime and try again

## Test Checklist

Before running full training:

- [ ] Local file structure test passes
- [ ] Colab environment setup works (cells 1-4)
- [ ] Drive mounts successfully
- [ ] Data setup completes (existing or downloaded)
- [ ] No import errors
- [ ] GPU is available (`torch.cuda.is_available()` returns `True`)

If all checks pass, proceed with full training!

## Expected Training Time

- **English model**: 30-60 minutes on Colab T4 GPU
- **Japanese model**: 1-2 hours on Colab T4 GPU

## Monitoring Training

Watch for:
- Loss decreasing over epochs
- No CUDA errors
- Checkpoints saving to Drive
- Model export completing successfully

## After Training

1. Check email for completion notification
2. Verify models in Drive: `/Phu's Data development/models/`
3. Download model zip files
4. Test on actual devices

---

**Pro Tip**: Run English model first (faster) to validate the entire workflow before running the longer Japanese training.
