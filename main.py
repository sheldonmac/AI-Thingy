import os

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize, rotate
from sklearn.svm import LinearSVC

# Settings
IMAGE_SIZE = (128, 128)
DATASET_PATH = "images"
CATEGORIES = ["2x2", "3x2"]


def runtime():
    def load_images(augment=True):
        x = []
        y = []

        for label, category in enumerate(CATEGORIES):
            folder = os.path.join(DATASET_PATH, category)

            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)
                image = imread(img_path)
                image = rgb2gray(image)
                image = resize(image, IMAGE_SIZE)

                features = hog(
                    image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                )
                x.append(features)
                y.append(label)

        return np.array(x), np.array(y)

    # load images
    x, y = load_images(augment=True)
    model = LinearSVC()
    model.fit(x, y)

    print(f"Training completed on {len(x)} images (including augmented).")

    # run test image
    def predict_image(file_path):
        image = imread(file_path)
        image = rgb2gray(image)
        image = resize(image, IMAGE_SIZE)
        features = hog(
            image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
        )
        prediction = model.predict([features])[0]
        return CATEGORIES[prediction]

    # test user image
    test_file = input("path to file: ")
    return f"Prediction for {test_file}: {predict_image(test_file)}"


print(runtime())
