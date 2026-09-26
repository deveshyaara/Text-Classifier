import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
import keras
import time

INPUT_SIZE = 32
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

@keras.saving.register_keras_serializable(package="vit")
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@keras.saving.register_keras_serializable(package="vit")
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config


print("Loading ViT model...")
model = tf.keras.models.load_model("vit_model.keras")
print("ViT model loaded.")


def predict(img):
    if img is None:
        return {"error": "No image provided"}
        
    img = img.convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    
    t0 = time.perf_counter()
    logits = model(batch, training=False).numpy()[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    top_indices = np.argsort(probs)[::-1][:5].tolist()
    top_predictions = [
        {"class": CIFAR10_CLASSES[i], "probability": float(probs[i])}
        for i in top_indices
    ]
    
    predicted_idx = int(np.argmax(probs))
    
    return {
        "prediction": CIFAR10_CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "top_predictions": top_predictions,
        "all_probabilities": {cls: float(probs[i]) for i, cls in enumerate(CIFAR10_CLASSES)},
        "inference_time_ms": round(elapsed_ms, 2)
    }

# Create standard Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="ViT CIFAR-10 Model API",
    description="Backend microservice for AI Model Lab."
)

demo.launch(server_name="0.0.0.0", server_port=7860)

