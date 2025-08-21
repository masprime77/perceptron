import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
IMG_SIZE = 64
STEP_SIZE = 10
LABEL_MAP = {
    "circle": 1,
    "square": -1
}

w = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
b = 0.0

w_steps = []


def img_to_matrix(path: str) -> np.ndarray:
    """
    Reads an image from the given path and converts it to a matrix.
    ---
    path: str
        Path to the image file.
    Returns:
        np.ndarray: The image as a matrix of shape (IMG_SIZE, IMG_SIZE).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    if img.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"Image at {path} must be of shape ({IMG_SIZE}, {IMG_SIZE})")
    
    matrix = (img >= 128).astype(np.float32)
    
    return matrix


def classification(multiplication):
    """
    Classifies the image based on the dot product of the weights and the image matrix.
    ---
    multiplication: float
        The multiplier for the dot product.
    Returns:
        str: 1 if classified as a circle, -1 if classified as a square.
    """
    return "square" if multiplication > 0 else "circle"


def multiplication(x: np.ndarray):
    """
    Computes the dot product of the weights and the image matrix.
    ---
    x: np.ndarray
        The image matrix.
    Returns:
        The dot product result and the bias
    """
    return np.sum(w * x) + b


def update(true_label: int, predicted_label: str, x: np.ndarray):
    """
    Updates the weights and bias based on the true and predicted labels.
    ---
    true_label: str
        The true label of the image.
    predicted_label: str
        The predicted label of the image.
    x: np.ndarray
        The image matrix.
    """
    global w, b
    if true_label != predicted_label:
        if true_label == 1:
            w += x * w
            b += 1
        else:
            w -= x * w
            b -= 1

def calculate_weigth():
    """
    It reads images from the data directory, classifies them, and updates the weights.
    """
    steps = STEP_SIZE
    global w_steps

    tra_circle_dir = DATA_DIR / "train" / "circle"
    tra_square_dir = DATA_DIR / "train" / "square"
    
    circle_imgs = sorted(list(tra_circle_dir.glob("*.png")))
    square_imgs = sorted(list(tra_square_dir.glob("*.png")))

    for circle_path, square_path in zip(circle_imgs, square_imgs):
        x_c = img_to_matrix(str(circle_path))
        y_true_c = LABEL_MAP["circle"]
        y_pred_c = classification(multiplication(x_c))
        update(y_true_c, y_pred_c, x_c)

        x_s = img_to_matrix(str(square_path))
        y_true_s = LABEL_MAP["square"]
        y_pred_s = classification(multiplication(x_s))
        update(y_true_s, y_pred_s, x_s)

        if steps == 0:
            w_steps.append(w.copy())
            steps = STEP_SIZE
        else:
            steps -= 1

    print("Training completed. Weights and bias updated.")


def plot_weights():
    """
    Plots the weights at each step.
    """
    global w_steps
    index = 0

    for w in w_steps:
        vmax = np.max(np.abs(w))
        if vmax == 0:
            vmax = 1
        vmin = -vmax

        plt.figure(figsize=(4,4))
        plt.imshow(w, cmap="seismic", vmin=vmin, vmax=vmax)
        plt.colorbar(label="weights")
        plt.title("Weights Visualization")
        plt.axis("off")
        plt.savefig(f"data/w_images/weights_{index}.png")
        index += 1
        plt.close()


def main():
    calculate_weigth()
    plot_weights()


if __name__ == "__main__":
    main()
