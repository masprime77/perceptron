import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 64


def load_model(path):
    """
    Loads the model weights and bias from a file.
    ---
    path: str
        The path to the file containing the model weights and bias.
    Returns:
        tuple: A tuple containing the weights (np.ndarray) and bias (int).
    """
    w = np.zeros((2, 2), dtype=np.float32)
    b = 0.0

    base_path = Path(path)
    if not base_path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    for file in base_path.iterdir():
        if file.is_file() and file.suffix == '.npy':
            if file.name == 'weights.npy':
                w = np.load(file).astype(np.float32)
            elif file.name == 'bias.npy':
                b = float(np.load(file)[0])
            else:
                raise ValueError(f"Unexpected file in model directory: {file.name}")
    return w, b


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


def multiplication(x: np.ndarray, w, b):
    """
    Computes the dot product of the weights and the image matrix.
    ---
    x: np.ndarray
        The image matrix.
    Returns:
        The dot product result and the bias
    """
    return np.sum(w * x) + b


def test_model(w, b):
    for i in [1, 2, 3, 6, 7, 8, 9]:
        matrix_1 = img_to_matrix(f"data/train/circle/circle_00000{i}.png")
        if classification(multiplication(matrix_1, w, b)) == 1:
            print("The image is classified as a circle.\n"
            "Correct classification.")
        else:
            print("The image is classified as a square.\n"
            "Incorrect classification.")


def main():
    w, b = load_model("model/")
    print(b, w)
    test_model(w, b)
    

if __name__ == "__main__":
    main()
    