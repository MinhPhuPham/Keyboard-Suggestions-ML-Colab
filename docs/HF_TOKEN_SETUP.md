# Hugging Face Token Setup for Gemma-2

## Why You Need This

Gemma-2 is a "gated" model, which means you need to:
1. Accept the license agreement on Hugging Face
2. Authenticate with a token to download it

## Step-by-Step Setup

### 1. Accept Gemma-2 License
1. Go to https://huggingface.co/google/gemma-2-2b-it
2. Click **"Agree and access repository"**
3. Accept the license terms

### 2. Create Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Settings:
   - **Name**: `colab-training` (or any name you like)
   - **Type**: **Read** (not Write)
   - **Repositories**: Leave as "All"
4. Click **"Create token"**
5. **Copy the token** (you won't see it again!)

### 3. Add Token to Google Colab Secrets

**In your Colab notebook:**

1. Click the **🔑 key icon** in the left sidebar (Secrets)
2. Click **"+ Add new secret"**
3. Enter:
   - **Name**: `HF_TOKEN` (must be exactly this)
   - **Value**: Paste your token from step 2
4. Click **"Add secret"**
5. **Toggle ON** the "Notebook access" switch for `HF_TOKEN`

### 4. Restart and Run

1. **Runtime** → **Restart runtime**
2. Re-run all cells from the beginning
3. The authentication cell will now work!

## Verification

When you run the authentication cell, you should see:
```
✓ Logged in to Hugging Face
```

If you see an error, double-check:
- Token name is exactly `HF_TOKEN` (case-sensitive)
- Notebook access toggle is ON
- You accepted the Gemma-2 license
- Token has "Read" permission

## Security Notes

- ✅ Colab secrets are encrypted and secure
- ✅ Secrets are not shared between notebooks unless you enable access
- ✅ Never commit tokens to Git
- ❌ Don't paste tokens directly in notebook code

## Alternative: Use Ungated Model

If you prefer not to use tokens, you can switch to an ungated model:

```python
# Instead of Gemma-2-2B
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # No token needed
```

Qwen2.5-1.5B is:
- Smaller (1.5B vs 2B params)
- Faster
- No authentication required
- Still very good quality
