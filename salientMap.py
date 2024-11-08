import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

model = VGG16(weights='imagenet', include_top=True)
img_path = 'img.jpg'  # Path to the dog image

try:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    x = tf.convert_to_tensor(x, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x)
        class_idx = tf.argmax(preds[0])
        class_output = preds[:, class_idx]

    grads = tape.gradient(class_output, x)
    saliency_map = np.max(np.abs(grads[0]), axis=-1)

    plt.imshow(saliency_map, cmap='jet')
    plt.colorbar()
    plt.title("VGG16 Saliency Map")
    plt.show()

except FileNotFoundError:
    print(f"Error: Image file not found at '{img_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")
