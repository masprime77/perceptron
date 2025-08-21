import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
IMG_SIZE = 64
LABEL_MAP = {
    "circle": 1,
    "square": -1
}

step_size = 0
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
        int: 1 if classified as a circle, -1 if classified as a square.
    """
    if multiplication == 0:
        return 0
    elif multiplication > 0:
        return 1
    else:
        return -1


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


def update(true_label: int, predicted_label: int, x: np.ndarray):
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
            w += x
            b += 1
        else:
            w -= x
            b -= 1

        save_weights(model="step", size=100)


def save_weights(model: str = "step", size: int = 10):
    """
    Saves the current weights and bias to a file.
    """
    global step_size, w_steps, w
    if model == "step":
        if step_size == 0:
            w_steps.append(w.copy())
            step_size = size
        else:
            step_size -= 1
    
    elif model == "all":
        w_steps.append(w.copy())

    else:
        raise ValueError("Model must be 'step' or 'all'.")


def calculate_weigth():
    """
    It reads images from the data directory, classifies them, and updates the weights.
    """
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

    print("Training completed. Weights and bias updated.")


def save_weights_colormap(w_matrix: np.ndarray, base_path:str, full_path: str, cmap: str = "seismic"):
    """
    Saves the weights as an image with a colormap.
    ---
    w_matrix: np.ndarray
        The weight matrix to be saved.  
    path: str
        The path where the image will be saved.
    cmap: str
        The colormap to use for saving the image. Default is "seismic".
    """
    
    Path(base_path).mkdir(parents=True, exist_ok=True)
    vmax = float(np.max(np.abs(w_matrix)))
    if vmax == 0:
        vmax = 1.0
    plt.imsave(full_path, w_matrix, cmap=cmap, vmin=-vmax, vmax=vmax)


def main():
    calculate_weigth()
    for i in range(len(w_steps)):
        save_weights_colormap(w_steps[i], "data/w_images/", f"data/w_images/weights_step_{i}.png")
    


if __name__ == "__main__":
    main()
