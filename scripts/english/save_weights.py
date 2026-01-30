# scripts/step1_export.py
import tensorflow as tf
import numpy as np
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'models')
KERAS_FILE = os.path.join(MODELS_DIR, 'gru_model.keras')
NUMPY_FILE = os.path.join(MODELS_DIR, 'gru_weights.npz')

print(f"Loading {KERAS_FILE}...")
model = tf.keras.models.load_model(KERAS_FILE, compile=False)

print("Extracting weights...")
weights_list = model.get_weights()

print(f"Saving to {NUMPY_FILE}...")
np.savez(NUMPY_FILE, *weights_list)
print("✅ Step 1 Done. Weights exported.")