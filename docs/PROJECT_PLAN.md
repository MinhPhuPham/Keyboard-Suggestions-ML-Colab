---
description: Keyboard Suggestion Model Development Plan (TinyBERT + N-gram Hybrid)
---

# Keyboard Suggestion Model Development Plan

## Overview

This plan outlines the development of a hybrid keyboard suggestion system for iOS and Android, focusing on word completion, next-word prediction, typo correction, and gibberish detection. The system uses a static TinyBERT model for high-quality, context-aware suggestions and an updatable N-gram trie for fast personalization and efficiency.

The plan is divided into two phases:

- **Phase 1: Static Model Development** – Build and optimize the core ML model, integrate initial N-gram, and ensure low memory/RAM usage.
- **Phase 2: Updatable N-gram Integration** – Add on-device personalization via the N-gram component, maintaining the static model.

---

## Goals

| Metric | Target |
|:-------|:-------|
| **Latency** | 50-100ms per suggestion |
| **Model Size** | <10MB total (static model + N-gram resources) |
| **RAM Usage** | <50MB runtime on mid-range devices (e.g., iPhone 12 or Snapdragon 888 equivalents) |
| **Accuracy** | 85-90% top-3 recall |
| **Deployment** | iOS (CoreML in UIKeyboardExtension) and Android (TFLite in InputMethodService) |
| **Timeline** | 4-6 weeks (assuming 1 developer with Colab access) |

---

## Tech Stack

**Training:**
- PyTorch
- Hugging Face Transformers (Colab GPU)

**Optimization:**
- Quantization
- Pruning

**N-gram:**
- Custom trie (Python for build, Swift/Kotlin for runtime)

**Datasets:**
- Single English words (completion)
- WikiText-103 (next-word)
- CSV with actual vs. misspelled words (typo correction)
- All via Hugging Face/Kaggle/GitHub

**Testing:**
- iOS Simulator
- Android Emulator
- Tools: Instruments/Perfetto for profiling

---

## Dataset Structure

### Local Storage (`/datasets`)

| File | Purpose | Format |
|:-----|:--------|:-------|
| `single_word_freq.csv` | Single words with frequency for completion and N-gram building | `word,count_frequency` |
| `misspelled.csv` | Typo correction pairs | `number,correct_word,misspelled_words` |

### Cloud Storage (Google Drive: `/datasets`)

Same structure as local storage for backup and collaboration:
- `single_word_freq.csv`
- `misspelled.csv`

**Additional datasets** (to be downloaded during training):
- **WikiText-103**: From Hugging Face (`wikitext`) for next-word prediction and N-gram building
- **Google-10000-English**: From GitHub (optional, if `single_word_freq.csv` is insufficient)

---

## Phase 1: Static Model Development

Focus on building a lightweight, static TinyBERT model fine-tuned for the features. Integrate a base (non-updatable) N-gram trie early to optimize the hybrid flow for memory and RAM. This ensures the system runs efficiently from the start, with N-gram handling fast lookups and model providing quality boosts.

### 1.1 Model Selection and Architecture

**Base Model:**
- TinyBERT (4 layers, 312 hidden size, 4 heads, ~14M params)
- Pre-trained from Hugging Face (`google/bert_uncased_L-4_H-312_A-4`), then fine-tuned on your datasets

**Task Setup:**
- Multi-task Masked Language Modeling (MLM)
- **Input:** `"{context} [MASK]"` (e.g., "Hel [MASK]" for completion; "The quick brown [MASK]" for next-word; "buld [MASK]" for typo)
- **Output:** Top-k tokens (k=30), filtered to words

**Initial N-gram Integration:**
- Build a base 2-3 gram trie from WikiText corpus + `single_word_freq.csv` for prefix/bigram lookups
- Use it to rank/filter model outputs (e.g., model suggests candidates, N-gram scores them by frequency/probability)

**Hybrid Flow Optimization:**

Suggestion Pipeline:
1. User input → Check N-gram trie first (fast, <10ms, low RAM)
2. If matches <3 or confidence low, query static model (<50ms)
3. Combine: Re-rank model outputs with N-gram scores (e.g., log-prob fusion)
4. Output top-3 suggestions

**Memory/RAM Optimizations:**
- Trie compressed to <2MB (binary serialization)
- Model quantized to INT8 (<5MB)
- Total runtime RAM: <30MB (trie in-memory, model inference-only)

**Gibberish Detection:**
- Heuristic in flow – If N-gram returns no valid prefixes after 5 chars or model prob <0.1, flag as gibberish (entropy check via simple string analysis, <1ms)

### 1.2 Datasets and Preparation

**Sources:**

| Feature | Dataset | Location | Size |
|:--------|:--------|:---------|:-----|
| Completion | Single English words | `/datasets/single_word_freq.csv` (local/drive) | ~10k-50k words |
| Next-Word | WikiText-103 | Hugging Face: `wikitext` | ~100M words |
| Typo Correction | Misspelled pairs | `/datasets/misspelled.csv` (local/drive) | ~20k-50k entries |
| N-gram Build | WikiText + word frequency | WikiText corpus + `single_word_freq.csv` | - |

**Preparation Steps:**

1. **Completion:**
   - Load `single_word_freq.csv`
   - Generate ~50k samples by truncating words (e.g., "hello" → "hel [MASK]")
   - Use frequency counts for weighted sampling

2. **Next-Word:**
   - Download WikiText-103 from Hugging Face
   - Extract ~100k samples (split sentences into short contexts like "The quick [MASK]", label "brown")

3. **Typo Correction:**
   - Load `misspelled.csv`
   - Generate ~50k samples (e.g., "buld [MASK]" → "build")
   - Handle multiple misspellings per correct word

4. **Combine:**
   - Mix into JSONL (~200k total: 40% completion, 40% next-word, 20% typo)
   - Split: 90% train / 10% validation

5. **Tokenize:**
   - BERT tokenizer
   - `max_len=16`

6. **Build Base N-gram:**
   - Use KenLM or custom Python trie from WikiText + `single_word_freq.csv`
   - Export as serialized file

**Size Optimization:**
- Limit vocab to 10k common words for trie compression

### 1.3 Training and Fine-Tuning

**Setup:**
- Colab T4 GPU
- Batch size: 32
- Epochs: 3-5
- Optimizer: AdamW (lr=3e-5)

**Process:**
1. Load pre-trained TinyBERT
2. Fine-tune on mixed dataset (MLM loss)
3. Evaluate: Perplexity <5, top-3 accuracy >85% on validation set

**Optimization for Memory/RAM:**
- Add LoRA (r=8) on last 2 layers during fine-tuning (reduces param updates)
- Prune 20% weights (`torch.prune`)
- Test hybrid: Simulate inputs, profile RAM (e.g., Python with `memory_profiler`)

### 1.4 Optimization and Testing

**Techniques:**

| Optimization | Target | Method |
|:-------------|:-------|:-------|
| Quantization | Model <5MB | INT8 post-training (`coremltools`/`TFLiteConverter`) |
| Compression | Trie <2MB | Binary serialization |
| RAM Usage | <50MB runtime | Short sequences, batch=1 inference |

**Testing:**

- **Latency:** Benchmark 1000 inputs (aim <80ms avg)
- **Accuracy:** Custom metrics (top-3 recall per feature)
- **Devices:** iOS Simulator (Xcode), Android Emulator (Studio)
- **Edge Cases:** Low RAM scenarios (e.g., simulate with limits)

### 1.5 Export and Initial Deployment

**iOS (CoreML):**
```python
# Convert using coremltools
coremltools.convert(model, inputs=...)
```
- Bundle: Include model + base N-gram file in extension

**Android (TFLite):**
```bash
# Convert using optimum-cli
optimum-cli export tflite --model ...
```
- Bundle: Assets folder

**Hybrid Flow Validation:**
- Implement prototype apps
- Ensure RAM <50MB during typing

### Phase 1 Milestones

| Week | Tasks | Deliverables |
|:-----|:------|:-------------|
| 1-2 | Datasets preparation, training, base N-gram build | Prepared datasets, trained model, base N-gram |
| 3 | Optimization, hybrid flow implementation, testing | Optimized model files, base N-gram, prototype code |

---

## Phase 2: Updatable N-gram Integration with Static Model

Build on Phase 1 by making the N-gram component updatable on-device for personalization, while keeping the model static. This allows "learning" from user selections without ML retraining, optimizing for memory/RAM by using lightweight updates.

### 2.1 N-gram Design for Updatability

**Structure:**
- Prefix trie for completion/typo
- Bigram matrix for next-word

**Components:**
- **Base:** Static file from Phase 1 (built from WikiText + `single_word_freq.csv`)
- **Personal:** Small overlay file (~0.5-1MB, user-specific frequencies)

**Update Mechanism:**

1. On selection: Increment count for chosen word/bigram (e.g., +1 frequency)
2. Batch updates: Every 50-100 selections, merge personal into base (in-memory, <5ms)
3. Persistence: Save personal file to device storage (Documents/SharedPreferences)

**Hybrid Flow Update:**
- Suggestion: Use merged trie (base + personal) for ranking
- If user picks non-top suggestion, boost its personal frequency (e.g., x2 multiplier)

**Memory/RAM Optimization:**
- Personal trie limited to 5k entries (auto-prune old/low-freq)
- Runtime: Load once on app start (<20MB RAM total with model)

### 2.2 Implementation Details

**Cross-Platform Code:**

**iOS (Swift):**
- Use Trie data structure
- Update in background thread
- Store personal in NSUserDefaults or FileManager

**Android (Kotlin):**
- Custom Trie class
- Update via WorkManager
- Store in Room DB or files

**Features Enhancement:**
- **Completion/Typo:** Personal trie prioritizes user-frequent words
- **Next-Word:** Update bigram probabilities based on selections
- **Gibberish:** If repeated detections, add to personal "ignore" list

**Global Improvements:**
- Push new base N-gram via app updates or remote (e.g., Firebase; download <1MB files)

### 2.3 Testing and Optimization

**Personalization Tests:**
- Simulate 1000 user sessions
- Measure accuracy improvement (e.g., +10% after 200 selections)

**Memory/RAM:**
- Profile updates (aim <10MB spike)
- Limit updates to off-peak (e.g., app close)

**Edge Cases:**
- Handle storage limits
- Resets (e.g., clear personal on uninstall)

### 2.4 Final Deployment and Maintenance

**iOS:**
- Submit to App Store
- Extensions auto-update

**Android:**
- Play Store
- Use splits for smaller APKs

**Monitoring:**
- Add analytics (e.g., suggestion acceptance rate)

**Future-Proofing:**
- If needed, add simple rules (e.g., Levenshtein for typos) without increasing size

### Phase 2 Milestones

| Week | Tasks | Deliverables |
|:-----|:------|:-------------|
| 4 | Updatable N-gram implementation, integration | Updatable N-gram code |
| 5-6 | Testing, deployment prototypes | Full hybrid code, user simulation scripts |

---

## Risks and Mitigations

| Risk | Mitigation |
|:-----|:-----------|
| High RAM | Mitigate with pruning/limits; fallback to N-gram-only if detected |
| Accuracy Drops | Add more data in Phase 1; user feedback loop |
| App Rejection | Ensure privacy (no cloud for personal data); comply with guidelines |

---

## Resources Needed

- **Training:** Google Colab (free T4 GPU)
- **Datasets:** 
  - Local/Drive: `/datasets/single_word_freq.csv`, `/datasets/misspelled.csv`
  - Hugging Face: WikiText-103 (free)
  - GitHub: Google-10000-English (optional, free)
- **Developer Tools:** Xcode (iOS), Android Studio (Android)

---

This plan ensures an efficient, personalizable keyboard system optimized for mobile constraints as of 2026.