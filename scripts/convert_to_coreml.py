import tensorflow as tf
import coremltools as ct
import numpy as np
import os
from tensorflow.keras import layers, models

# --- 1. Setup Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'models')
KERAS_FILE = os.path.join(MODELS_DIR, 'gru_model.keras')
NUMPY_FILE = os.path.join(MODELS_DIR, 'gru_weights.npz')
COREML_MODEL = os.path.join(MODELS_DIR, 'gru_keyboard_ios.mlpackage')

# print(f"TensorFlow Version: {tf.__version__}")

# print(f"Loading {KERAS_FILE}...")
# model = tf.keras.models.load_model(KERAS_FILE, compile=False)

# print("Extracting weights...")
# weights_list = model.get_weights()

# print(f"Saving to {NUMPY_FILE}...")
# np.savez(NUMPY_FILE, *weights_list)
# print("✅ Step 1 Done. Weights exported.")

# --- 2. Rebuild Architecture ---
print("[1/3] Rebuilding Model Architecture...")

def build_model():
    # Input: Sequence of 10 integers
    inputs = layers.Input(shape=(10,), dtype='float32', name='input_layer')
    
    x = layers.Embedding(input_dim=25001, output_dim=128)(inputs)
    x = layers.GRU(256, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(25001, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

model = build_model()

# --- 3. Load Weights ---
print(f"[2/3] Loading Weights from {NUMPY_FILE}...")
try:
    with np.load(NUMPY_FILE) as data:
        weights = [data[k] for k in data.files]
        model.set_weights(weights)
    print("      Weights injected successfully.")
except Exception as e:
    print(f"❌ Weight Error: {e}")
    exit(1)

# --- 4. Convert to CoreML ---
print("[3/3] Converting to CoreML...")

try:
    # Convert
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        inputs=[
            ct.TensorType(
                name="input_layer", 
                shape=(1, 10),      
                dtype=np.int32      # iOS Input is Integers
            )
        ]
        # We removed the manual 'outputs' and 'rename' logic to prevent errors.
        # CoreML will name the output automatically (usually 'Identity').
    )

    # Metadata
    mlmodel.short_description = "GRU_Keyboard_Suggestions"
    mlmodel.author = "MinhPhuPham"
    mlmodel.version = "1.0"
    
    # SAVE THE MODEL
    mlmodel.save(COREML_MODEL)
    
    print("\n" + "═"*70)
    print(f"SUCCESS! Model saved to: {COREML_MODEL}")
    print("═"*70)
    print("👉 Open this file in Xcode to see your Output Name (usually 'Identity')")

except Exception as e:
    print("\n❌ Conversion Failed:", str(e))