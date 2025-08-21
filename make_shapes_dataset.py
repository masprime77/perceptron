import os
from pathlib import Path
import numpy as np
import cv2

SEED = 11
IMG_SIZE = 64
OUT_DIR = Path("data")
SPLITS = ["train", "val", "test"]
SHAPES = ["circle", "square"]
MIN_R, MAX_R = 6, 24
MIN_S, MAX_S = 10, 40
N_PER_SHAPE = 500
SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test


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


def random_cicle_params():
    """
    Generates random parameters for a circle.
    ---
    Returns:
        tuple: (cx, cy, r) where cx and cy are the center coordinates and r is the radius.
    """
    r = np.random.randint(MIN_R, MAX_R + 1)
    cx = np.random.randint(r, IMG_SIZE - r)
    cy = np.random.randint(r, IMG_SIZE - r)
    return cx, cy, r


def random_square_params():
    """
    Generates random parameters for a square.
    ---
    Returns:
        tuple: (x0, y0, size) where x0 and y0 are the top-left corner coordinates and size is the side length.
    """
    size = np.random.randint(MIN_S, MAX_S + 1)
    x0 = np.random.randint(0, IMG_SIZE - size + 1)
    y0 = np.random.randint(0, IMG_SIZE - size + 1)
    return x0, y0, size


def make_one_image(shape: str, filled: bool = True) -> np.ndarray:
    """
    Creates a single image with a shape in a black canvas.
    ---
    shape: str
        The type of shape to draw ('circle' or 'square').
    filled: bool
        Whether the shape should be filled or not.
    Returns:
        np.ndarray: The generated image with the shape drawn on it.
    """
    img = create_black_canvas()
    if shape == "circle":
        cx, cy, r = random_cicle_params()
        draw_circle(img, cx, cy, r, filled)
    elif shape == "square":
        x0, y0, size = random_square_params()
        draw_square(img, x0, y0, size, filled)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return img


def comptute_split_counts(n: int, ratios):
    """
    Computes the number of samples for each split based on the given ratios.
    ---
    n: int
        Total number of samples.
    ratios: list of float
        Ratios for each split (e.g., [0.7, 0.15, 0.15] for train, val, test).
    Returns:
        list of int: Number of samples for each split.
    """
    counts = [int(n * ratio) for ratio in ratios]
    return counts


def make_filename(shape: str, idx: int) -> str:
    """
    Creates a filename for the image based on the shape and index.
    ---
    shape: str
        The type of shape ('circle' or 'square').
    idx: int
        The index of the image.
    Returns:
        str: The filename in the format "shape_idx.png".
    """
    return f"{shape}_{idx:06d}.png"


def populate_shape(shape: str, n_per_class: int):
    """
    Populates the dataset with images of a specific shape.
    ---
    shape: str
        The type of shape ('circle' or 'square').
    n_per_class: int
        Number of images to create.
    """
    assert shape in SHAPES, f"Unknown shape: {shape}"
    counts = comptute_split_counts(n_per_class, SPLIT_RATIO)

    index = np.arange(n_per_class)
    rng = np.random.default_rng(SEED + 0 if shape == "circle" else 1)
    rng.shuffle(index)

    i_tr_end = counts[0]
    i_va_end = i_tr_end + counts[1]

    idx_train = index[:i_tr_end]
    idx_val = index[i_tr_end:i_va_end]
    idx_test = index[i_va_end:]

    def _save_batch(split: str, idxs: np.ndarray):
        for idx in idxs:
            img = make_one_image(shape)
            filename = make_filename(shape, idx)
            save_png_gray(img, OUT_DIR / split / shape / filename)


    _save_batch("train", idx_train)
    _save_batch("val", idx_val)
    _save_batch("test", idx_test)

    print(f"[{shape}] -> train:{len(idx_train)}  val:{len(idx_val)}  test:{len(idx_test)}")


def build_dataset(n_per_class: int = N_PER_SHAPE):
    """
    Builds the dataset by populating it with circles and squares.
    ---
    n_per_class: int
        Number of images to create for each shape.
    """
    ensure_dirs(OUT_DIR)
    print(f"Building dataset with {n_per_class} images per shape...")

    for shape in SHAPES:
        populate_shape(shape, n_per_class)

    print("Dataset creation completed at: ", OUT_DIR.resolve())


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


def sanity_check_3():
    """
    Performs a sanity check to ensure the dataset can be created with random shapes.
    We will create 3 circles and 3 squares with random parameters.
    """
    for i in range(3):
        circle = make_one_image("circle")
        path = OUT_DIR / "temp" / f"circle_{i}.png"
        save_png_gray(circle, path)

    for i in range(3):
        square = make_one_image("square")
        path = OUT_DIR / "temp" / f"square_{i}.png"
        save_png_gray(square, path)
    
    print("Sanity check for random shapes passed.")


def main():
    np.random.seed(SEED)
    ensure_dirs(OUT_DIR)
    print(f"Creating dataset in: {OUT_DIR.resolve()}")
    # sanity_check_1()
    # sanity_check_2()
    # sanity_check_3()
    build_dataset(N_PER_SHAPE)


if __name__ == "__main__":
    main()