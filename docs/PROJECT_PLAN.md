Plan to Build Keyboard Suggestion Models (English and Japanese)
This plan combines our previous discussions, including the best model recommendations for each language (English: Microsoft Phi-3 Mini; Japanese: Alibaba Qwen3-14B or its 0.6B variant for mobile optimization). We'll build separate models for optimal performance, size (20–30 MB English, 40–60 MB Japanese post-optimization), and latency (<50–100 ms predictions), as established. The focus is on next-word/morpheme suggestions for a mobile keyboard (e.g., completing "Today is a day better than..." with "tomorrow").
The plan is divided into two sections (one per model), with shared steps where possible (e.g., optimizations). It aligns with your repo structure (e.g., train_english.py in docs/common) and phases (Phase 1: English; Phase 2: Japanese). Timeline: 1–2 weeks per model for MVP (assuming access to GPU like Google Colab free tier). Tools: Python/PyTorch/Hugging Face for training; ONNX for export.
Shared Prerequisites and Tools

Environment Setup: Use Google Colab or local Python 3.12+ with pip install torch transformers datasets fugashi unidic sentencepiece peft accelerate. Download UniDic for Japanese: python -m unidic download.
Data: As finalized—English: SwiftKey Corpus (~200 MB subset from Coursera-SwiftKey.zip); Japanese: CC100 Japanese Subset (1–2 GB streamed via Hugging Face).
Optimizations (Apply to Both):
Quantization: Use 4-bit (via bitsandbytes or Hugging Face quantize_model)—reduces size 4x with <2% accuracy drop.
Pruning: 20–30% weights removed (Torch nn.utils.prune).
LoRA for On-Device Learning: Add adapters (1–5 MB) for daily personalization (retrain on user selections, cap at 1000 pairs in SQLite with weekly cleanup).
Evaluation: Perplexity <20 on 10% held-out data; top-3 accuracy >85% (test with custom scripts on phrases like your example).
Export: To ONNX (torch.onnx.export), then Core ML (iOS) or LiteRT (Android).

Hardware: Train on Colab GPU (free tier sufficient for small models); inference tests on emulators (Xcode for iOS, Android Studio).
Common Code Base: Reuse a template script (e.g., in docs/common/base_train.py) for loading, fine-tuning, and exporting.

Plan to Build English Suggestion Model (Phase 1: 1 Week)
Focus: Word-level predictions optimized for casual English (e.g., slang, time phrases). Model: Microsoft Phi-3 Mini (3.8B params, pruned/quantized to ~100–200M for mobile)—best for English due to strong causal LM (next-token) performance on conversational data (85–92% top-3 accuracy).

Data Preparation (1–2 Days):
Download SwiftKey Corpus ZIP; extract en_US folder (focus on twitter.txt and blogs.txt for ~200 MB subset).
Clean: Lowercase, remove duplicates/URLs (use NLTK: word_tokenize(text.lower())).
Augment: Add 5–10% emojis/special chars (e.g., from EmoTag dataset via Hugging Face: load_dataset('badrex/EmoTag')—mix sentences like "happy day 😊").
Split: 80% train, 10% val, 10% test (~5M tokens total).
Tokenize: NLTK + SentencePiece BPE (cap vocab 15–25k: spm.SentencePieceTrainer.train(input='train.txt', vocab_size=20000)).
Store: In data/english/ as train.txt, val.txt.

Model Setup and Fine-Tuning (2–3 Days):
Load Pre-Trained: from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct'); tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct').
Add LoRA: from peft import LoraConfig, get_peft_model; config = LoraConfig(r=8, target_modules=['q_proj', 'v_proj']); model = get_peft_model(model, config).
Fine-Tune: Use Hugging Face Trainer (sequence length=8, batch=32, epochs=3–5, lr=1e-5). Objective: Causal LM (next-token loss). Code snippet:Pythonfrom datasets import load_dataset
dataset = load_dataset('text', data_files={'train': 'data/english/train.txt'})
trainer = Trainer(model=model, train_dataset=dataset['train'], args=TrainingArguments(output_dir='english_model', num_train_epochs=3))
trainer.train()
Handle Features: Gibberish detection (entropy check in inference code); swipe prep (rerank candidates).

Optimization and Export (1–2 Days):
Prune: import torch.nn.utils.prune as prune; prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=0.3).
Quantize: model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).
Test Size/Latency: Aim 20–30 MB (os.path.getsize('english_model.pt')); <50 ms on emulator.
Export: ONNX (torch.onnx.export(model, torch.randn(1,8), 'models/android/english_model.onnx')); convert to Core ML via coremltools for iOS.
Integrate Learning: Add LoRA retrain function (nightly on user data).

Testing and Iteration (1 Day):
Eval: Perplexity on val set; manual tests (e.g., "Today is a day better than..." → check "tomorrow" in top-3).
Deploy Prototype: In tests/ios and tests/android, simulate keyboard input.
Save: In models/ios/english_model.mlmodel and models/android/english_model.onnx.


Plan to Build Japanese Suggestion Model (Phase 2: 1–2 Weeks)
Focus: Morpheme/kanji-level predictions for IME (e.g., romaji input → kanji suggestions). Model: Alibaba Qwen3-14B (or Qwen3-0.6B for mobile; pruned/quantized to ~200–500M)—best for Japanese due to multilingual excellence (80–90% accuracy on JGLUE, handles kanji/topics like "良い日" → "明日").

Data Preparation (2–3 Days):
Load CC100 Subset: Stream 10–20% (~1–2 GB) via ds = load_dataset('cc100', lang='ja', split='train[:10%]').
Clean: Remove non-Japanese/URLs (use regex); no lowercase.
Augment: Mix 5–10% emojis (Japanese EmoTag subsets); add romaji/kanji pairs from JMDict (download: ftp://ftp.edrdg.org/pub/Nihongo/JMdict.gz).
Split: 80% train, 10% val (use AJIMEE-Bench for test: GitHub azookey/ajimee-bench).
Tokenize: fugashi for morphemes (tagger = fugashi.Tagger('-Owakati')) + SentencePiece BPE (cap vocab 40–60k).
Store: In data/japanese/ as train.jsonl (for streaming).

Model Setup and Fine-Tuning (3–5 Days):
Load Pre-Trained: model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct'); tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct') (use 1.5B variant for balance if 14B too heavy).
Add LoRA: Same as English (target Q/K/V projections).
Fine-Tune: Trainer with sequence length=8, batch=16 (due to size), epochs=3–5, lr=5e-6. Include kanji conversion (prompt as "romaji: [input] → kanji: [prediction]").Pythonfrom datasets import load_dataset
dataset = load_dataset('json', data_files={'train': 'data/japanese/train.jsonl'})
trainer = Trainer(model=model, train_dataset=dataset['train'], args=TrainingArguments(output_dir='japanese_model', num_train_epochs=3))
trainer.train()
Handle Features: Adapt gibberish for kana (e.g., repeat checks); swipe with flick layout (rerank romaji candidates).

Optimization and Export (2–3 Days):
Prune: Same as English (amount=0.3–0.4 for larger model).
Quantize: 4-bit (model.quantize(bits=4, method='awq') via bitsandbytes).
Test Size/Latency: Aim 40–60 MB; <80 ms (Qwen's efficiency shines here).
Export: ONNX, then platform-specific (add IME layer for kanji ranking via small dict).
Integrate Learning: LoRA for user-preferred kanji (e.g., retrain on selections).

Testing and Iteration (1–2 Days):
Eval: Perplexity on val; AJIMEE-Bench for IME accuracy; tests like "今日は昨日より良い日だ..." → "明日".
Deploy Prototype: Simulate IME in tests.
Save: In models/ios/japanese_model.mlmodel and models/android/japanese_model.onnx.