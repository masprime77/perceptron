import os
from pathlib import Path
import numpy as np
import cv2

SEED = 11
IMG_SIZE = 64
OUT_DIR = Path("data")
SPLITS = ["train", "val", "test"]
SHAPES = ["circle", "square"]


def ensure_dirs(base: Path):
    """
    Ensures that the direction is correctly set up.
    ---
    base: str
        Path to the base directory.
    """
    for split in SPLITS:
        for shape in SHAPES:
            path = base / split / shape
            path.mkdir(parents=True, exist_ok=True)


def save_png_gray(img:np.ndarray, path: Path):
    """
    Saves a grayscale image as PNG.
    ---
    img: np.uint8, shape(64, 64)
        The image to save.
    path: Path
        The path to save the image to.
    """
    assert img.dtype == np.uint8
    assert img.ndim == 2 and img.shape == (IMG_SIZE, IMG_SIZE)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to save image to {path}")
    

def create_black_canvas():
    """
    Creates a black canvas of size IMG_SIZE x IMG_SIZE.
    ---
    Returns:
        np.ndarray: A black canvas of shape (IMG_SIZE, IMG_SIZE) with dtype np.uint8.
    """
    return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

def draw_circle(img, cx, cy, r, filled=True):
    """
    Draws a circle on the image.
    ---
    img: np.ndarray
        The image to draw on.
    cx: int
        The x-coordinate of the circle's center.
    cy: int
        The y-coordinate of the circle's center.
    r: int
        The radius of the circle.
    filled: bool
        Whether to fill the circle or not.
    """
    thickness = -1 if filled else 1
    cv2.circle(img, (cx, cy), r, color=255, thickness=thickness)


def draw_square(img, x0, y0, size, filled=True):
    """
    Draws a square on the image.
    ---
    img: np.ndarray
        The image to draw on.
    x0: int
        The x-coordinate of the top-left corner of the square.
    y0: int
        The y-coordinate of the top-left corner of the square.
    size: int
        The size of the square (one side).
    filled: bool
        Whether to fill the square or not.
    """
    pt1 = (x0, y0)
    pt2 = (x0 + size, y0 + size)
    thickness = -1 if filled else 1
    cv2.rectangle(img, pt1, pt2, color=255, thickness=thickness)

def sanity_check_1():
    """
    Performs a sanity check to ensure the dataset directory structure is correct.
    """
    img = create_black_canvas()
    test_path = OUT_DIR / "temp" / "sanity_check.png"
    save_png_gray(img, test_path)
    print(f"Sanity check passed, test image saved at {test_path.resolve()}")


def sanity_check_2():
    """
    Performs a sanity check to ensure the dataset can be created with basic shapes.
    """
    img1 = create_black_canvas()
    draw_circle(img1, 32, 32, 20, filled=True)
    save_png_gray(img1, OUT_DIR / "temp" / "circle.png")
    
    img2 = create_black_canvas()
    draw_square(img2, 10, 10, 40, filled=True)
    save_png_gray(img2, OUT_DIR / "temp" / "square.png")
    
    print("Sanity check for shapes passed.")


def main():
    np.random.seed(SEED)
    ensure_dirs(OUT_DIR)
    print(f"Creating dataset in: {OUT_DIR.resolve()}")
    sanity_check_1()
    sanity_check_2()


if __name__ == "__main__":
    main()