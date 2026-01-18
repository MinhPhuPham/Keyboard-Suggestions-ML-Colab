---
description: Keyboard Suggestion Model Agent (English Focus, with Japanese Expansion)
---

# Request Handler Workflow

When receiving a user request related to the keyboard suggestion project (TinyBERT static model + N-gram hybrid for features like word completion, next-word prediction, typo correction, and gibberish detection), follow this process:

---

## Step 1: Classify the Task

Identify which of the 4 categories the request belongs to, tailored to the project (English model primary; Japanese similar in future):

| Icon | Type        | Keywords to Detect |
|:----:|:------------|:-------------------|
| 🔍 | **CONSULT** | "should", "recommend", "compare", "suggest", "advice", "dataset choice", "model selection" |
| 🏗️ | **BUILD**   | "create", "make", "build", "add", "implement", "write", "fine-tune", "integrate N-gram" |
| 🔧 | **DEBUG**   | "error", "bug", "not working", "wrong", "fix", "latency issue", "accuracy drop" |
| ⚡ | **OPTIMIZE** | "slow", "refactor", "clean", "improve", "optimize", "reduce size", "lower latency" |

> **Note:** If unclear or involves Japanese expansion (e.g., adapting English model) → Ask the user before proceeding. Treat Japanese as parallel to English (same structure, datasets, etc.).

---

## Step 2: Execute Based on Mode

### 🔍 CONSULT Mode

1. Clarify context & constraints (e.g., mobile latency <100ms, size <10MB, English vs. Japanese datasets)
2. Provide 2-3 options with clear trade-offs (e.g., TinyBERT vs. BERT-Tiny for model size/accuracy)
3. Recommend the optimal option with reasoning (e.g., prioritize pre-trained + fine-tuning)
4. **WAIT for confirmation** before coding or dataset processing

### 🏗️ BUILD Mode

1. Confirm scope & acceptance criteria (e.g., features: completion/next-word/typo/gibberish; datasets: single words/WikiText/CSV typos)
2. Propose file/component structure (e.g., model training script, N-gram builder, export for CoreML/TFLite)
3. Build in order: Datasets → Model Fine-Tuning → Hybrid Integration → Export/Deployment
4. Run checklist before delivery; ensure compatibility for future Japanese (e.g., multilingual tokenizers)

### 🔧 DEBUG Mode

1. Gather info: what (e.g., low accuracy), where (e.g., next-word prediction), when (e.g., on iOS)
2. Analyze root cause (e.g., dataset mismatch in WikiText formality)
3. Propose fix + explanation (e.g., augment datasets with conversational sources)
4. Suggest prevention measures (e.g., add unit tests for latency/accuracy)

### ⚡ OPTIMIZE Mode

1. Measure baseline (e.g., current latency/size via profiling)
2. Identify main bottlenecks (e.g., model inference, N-gram lookups)
3. Propose improvements + predict results (e.g., INT8 quantization → <5MB size, 20% faster)
4. Refactor + compare before/after (e.g., test on simulators/emulators)

---

## Step 3: Pre-Delivery Checklist

**Code Quality:**
- [ ] No `any` types
- [ ] No hardcoded magic numbers/strings (e.g., use constants for latency targets like 100ms)
- [ ] Proper error handling (e.g., dataset loading failures)
- [ ] Clear variable/function naming (e.g., `fine_tune_tinybert` instead of vague names)

**Structure:**
- [ ] Correct folder structure (e.g., datasets in `/data/`, models in `/models/`)
- [ ] Consistent naming convention (e.g., snake_case for Python)
- [ ] Split files appropriately (< 200 lines/file; e.g., separate data prep from training)

**Project-Specific (ML/Mobile):**
- [ ] Datasets validated (single words for completion, WikiText extracts for next-word, CSV for typos)
- [ ] Model optimized for mobile (latency <100ms, size <10MB post-quantization)
- [ ] Hybrid flow tested (N-gram first, model fallback)
- [ ] Gibberish heuristic implemented (entropy-based, no ML)

**Documentation:**
- [ ] No new document files created outside structure. Update/extract content from existing docs and add to `/docs/english/` folder only.
- [ ] Use these specific files (create if missing, but update primarily):
  - **ENGLISH_ML.md**: For English keyboard model details (e.g., TinyBERT fine-tuning, datasets like WikiText/single words/CSV typos, hybrid N-gram integration).
  - **TESTING_GUIDE.md**: For testing purposes and related topics (e.g., accuracy metrics, latency benchmarks, unit tests for features).
  - **IOS_INTEGRATION.md**: For iOS-specific documentation (e.g., CoreML export, keyboard extension setup, deployment steps).
  - **ANDROID_INTEGRATION.md**: For Android-specific documentation (e.g., TFLite export, InputMethodService integration, deployment steps).
- [ ] For future Japanese: Mirror English structure (e.g., add `/docs/japanese/` with similar files like JAPANESE_ML.md; treat as identical to English model workflow, adapting datasets/tokenizers as needed).

---

## Tips

- ❌ Don't expand scope unilaterally (e.g., add Japanese without confirmation)
- ❌ Don't use `any` types
- ✅ Ask when requirements are unclear (e.g., dataset sources beyond WikiText/CSV)
- ✅ Comment complex logic (e.g., MLM input formatting)
- ✅ Prioritize: Readability → Performance → Cleverness
- ✅ For multi-language: Design with extensibility (e.g., language-agnostic code; English as base, Japanese as variant)
