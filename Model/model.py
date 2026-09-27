from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os
import glob

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Folder containing images to classify
IMAGE_FOLDER = "../Testing"

# Supported extensions
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

image_paths = sorted(
    p for p in glob.glob(os.path.join(IMAGE_FOLDER, "*"))
    if p.lower().endswith(valid_exts)
)

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")

# Preallocate the batch array: (N, 224, 224, 3)
size = (224, 224)
data = np.ndarray(shape=(len(image_paths), 224, 224, 3), dtype=np.float32)

# Load and preprocess every image into the batch array
for i, image_path in enumerate(image_paths):
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[i] = normalized_image_array

# Single batched prediction call
predictions = model.predict(data, verbose=0)

# predictions.shape == (N, num_classes)
results = []
for image_path, prediction in zip(image_paths, predictions):
    index = np.argmax(prediction)
    class_name = class_names[index].strip()[2:]
    confidence_score = prediction[index]

    results.append((image_path, class_name, confidence_score))

    print(f"{os.path.basename(image_path)} -> Class: {class_name}  "
          f"Confidence Score: {confidence_score:.4f}")