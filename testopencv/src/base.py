import os
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    'C:\\Dev\\Python\\Plain\\face-identifier-detector\\testopencv\src\\cascades\\data\\haarcascade_frontalface_alt2.xml')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

x_train = []
y_labels = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(
                path)).replace("-", " ").lower()
            print(label, path)
            # y_labels.append(label)
            # x_train.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            print(image_array)
