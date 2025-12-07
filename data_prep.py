import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
import csv
import sys
import atexit
import select
from pathlib import Path
from typing import Any, cast

import cv2
import pandas as pd

try:
    import msvcrt  # Windows-only, used for non-blocking ESC detection
except ImportError:
    msvcrt = None

try:
    import termios as _termios
    import tty as _tty
except ImportError:
    _termios = None
    _tty = None

termios: Any | None = cast(Any, _termios)
tty: Any | None = cast(Any, _tty)

from tqdm import tqdm

from retinaface import RetinaFace
from utils.data_utils import load_image_boxes_from_csv, split_dataframe_by_images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

_NONBLOCKING_FD: int | None = None
_ORIGINAL_TERM_SETTINGS: list[int] | None = None


def _resolve_path(base_dir: Path, path_str: str | None) -> Path | None:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def enable_nonblocking_stdin() -> None:
    """Put stdin into cbreak mode so we can detect ESC on POSIX systems."""
    global _NONBLOCKING_FD, _ORIGINAL_TERM_SETTINGS

    if msvcrt is not None or _NONBLOCKING_FD is not None:
        return

    if not sys.stdin.isatty() or termios is None or tty is None:
        return

    fd = sys.stdin.fileno()
    _ORIGINAL_TERM_SETTINGS = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    _NONBLOCKING_FD = fd

    def _restore_terminal() -> None:
        global _NONBLOCKING_FD, _ORIGINAL_TERM_SETTINGS
        if (
            _NONBLOCKING_FD is not None
            and _ORIGINAL_TERM_SETTINGS is not None
            and termios is not None
        ):
            termios.tcsetattr(_NONBLOCKING_FD, termios.TCSADRAIN, _ORIGINAL_TERM_SETTINGS)
        _NONBLOCKING_FD = None
        _ORIGINAL_TERM_SETTINGS = None

    atexit.register(_restore_terminal)


def list_images_in_directory(image_dir: Path) -> list[str]:
    files = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path.relative_to(image_dir).as_posix())
    return files


def run_retinaface_detector(
    model,
    image_bgr,
    threshold: float,
    allow_upscaling: bool,
) -> list[tuple[float, float, float, float, float]]:
    """Run the serengil/retinaface detector and return [ymin, xmin, ymax, xmax, score] tuples."""
    faces = RetinaFace.detect_faces(
        img_path=image_bgr,
        threshold=threshold,
        model=model,
        allow_upscaling=allow_upscaling,
    )

    detections: list[tuple[float, float, float, float, float]] = []
    if not isinstance(faces, dict):
        return detections

    for face in faces.values():
        bbox = face.get("facial_area")
        score = float(face.get("score", 0.0))
        if not bbox or len(bbox) != 4 or score < threshold:
            continue

        x_min, y_min, x_max, y_max = map(float, bbox)
        if x_max <= x_min or y_max <= y_min:
            continue

        detections.append((y_min, x_min, y_max, x_max, score))

    return detections


def detect_boxes_for_path(
    image_rel_path: str,
    image_root: Path,
    retinaface_model,
    threshold: float,
    allow_upscaling: bool,
) -> list[tuple[int, int, int, int, float]]:
    full_image_path = image_root / Path(image_rel_path)
    if not full_image_path.exists():
        print(f"Warning: Image not found: {full_image_path}")
        return []

    img = cv2.imread(str(full_image_path))
    if img is None:
        print(f"Warning: Failed to load image: {full_image_path}")
        return []

    detections = run_retinaface_detector(
        retinaface_model,
        img,
        threshold,
        allow_upscaling=allow_upscaling,
    )

    ih, iw, _ = img.shape
    processed: list[tuple[int, int, int, int, float]] = []

    for ymin, xmin, ymax, xmax, score in detections:
        x1 = max(0, min(iw - 1, int(round(xmin))))
        y1 = max(0, min(ih - 1, int(round(ymin))))
        x2 = max(0, min(iw - 1, int(round(xmax))))
        y2 = max(0, min(ih - 1, int(round(ymax))))

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        if width == 0 or height == 0:
            continue

        reduced_height = max(1, int(round(height * 0.9)))
        y1 = max(0, y2 - reduced_height)
        height = y2 - y1
        if width == 0 or height == 0:
            continue

        target_width = height
        if width < target_width:
            pad = target_width - width
            left_pad = pad // 2
            right_pad = pad - left_pad
            x1 = max(0, x1 - left_pad)
            x2 = min(iw - 1, x2 + right_pad)
            width = x2 - x1
            if width <= 0:
                continue

        if width == 0 or height == 0:
            continue

        processed.append((x1, y1, width, height, score))

    return processed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RetinaFace detections and save them into a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--csv", "-c",
        type=str,
        default=None,
        help="Path to CSV file with image annotations (defaults to data/splits/train.csv if --image-dir is not set)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/raw/blazeface/",
        help="Directory containing images to scan when CSV annotations are not provided"
    )
    parser.add_argument(
        "--data-root", "-d",
        type=str,
        default="data/raw/blazeface",
        help="Root directory for image paths (prepended to CSV image_path)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.9,
        help="Detection score threshold"
    )
    parser.add_argument(
        "--allow-upscaling",
        action="store_true",
        help="Allow RetinaFace to upscale smaller images during preprocessing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory for generated CSVs when scanning a directory"
    )
    parser.add_argument(
        "--master-name",
        type=str,
        default="retinaface_master.csv",
        help="Filename of the master CSV generated from --image-dir"
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="train.csv",
        help="Train split filename when splitting detections from --image-dir"
    )
    parser.add_argument(
        "--val-name",
        type=str,
        default="val.csv",
        help="Validation split filename when splitting detections from --image-dir"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images to place in the validation split when using --image-dir"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for directory-based train/val split"
    )
    return parser.parse_args()



def esc_pressed() -> bool:
    """Check if ESC key was pressed without blocking the processing loop."""
    if msvcrt:
        while msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b"\x1b":
                return True
            # Consume the second byte for special keys (e.g., arrows)
            if key in {b"\x00", b"\xe0"} and msvcrt.kbhit():
                msvcrt.getch()
        return False

    if _NONBLOCKING_FD is None:
        return False

    try:
        ready, _, _ = select.select([_NONBLOCKING_FD], [], [], 0)
    except (OSError, ValueError):
        return False

    if not ready:
        return False

    try:
        key = os.read(_NONBLOCKING_FD, 1)
    except OSError:
        return False

    return key in {b"\x1b", b"q", b"Q"}


def run_csv_mode(
    csv_path: Path,
    data_root: Path,
    retinaface_model,
    threshold: float,
    allow_upscaling: bool,
) -> None:
    print(f"Loading CSV: {csv_path}")
    image_paths, _ = load_image_boxes_from_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")

    seen: set[str] = set()
    unique_image_paths: list[str] = []
    for image_path in image_paths:
        if image_path in seen:
            continue
        seen.add(image_path)
        unique_image_paths.append(image_path)

    new_csv_path = csv_path.with_name(csv_path.stem + "_new" + csv_path.suffix)
    csv_file = new_csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_path", "x1", "y1", "w", "h"])
    print(f"Writing filtered detections to {new_csv_path}")

    try:
        with tqdm(total=len(unique_image_paths), desc="Processing images", unit="img") as progress:
            for image_path in unique_image_paths:
                boxes = detect_boxes_for_path(
                    image_path,
                    data_root,
                    retinaface_model,
                    threshold,
                    allow_upscaling,
                )
                for x1, y1, width, height, _ in boxes:
                    csv_writer.writerow([image_path, x1, y1, width, height])

                progress.update(1)

                if esc_pressed():
                    print("ESC detected, stopping image processing loop.")
                    break
    finally:
        csv_file.close()

    print("CSV processing complete.")


def run_directory_mode(
    image_dir: Path,
    retinaface_model,
    threshold: float,
    allow_upscaling: bool,
    output_dir: Path,
    master_name: str,
    train_name: str,
    val_name: str,
    val_fraction: float,
    seed: int,
) -> None:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = list_images_in_directory(image_dir)
    if not image_paths:
        print(f"No image files found under {image_dir}")
        return

    print(f"Scanning {len(image_paths)} images from {image_dir}")

    master_rows: list[tuple[str, int, int, int, int]] = []
    dedup_keys: set[tuple[str, int, int, int, int]] = set()

    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as progress:
        for image_path in image_paths:
            boxes = detect_boxes_for_path(
                image_path,
                image_dir,
                retinaface_model,
                threshold,
                allow_upscaling,
            )
            for x1, y1, width, height, _ in boxes:
                key = (image_path, x1, y1, width, height)
                if key in dedup_keys:
                    continue
                dedup_keys.add(key)
                master_rows.append(key)

            progress.update(1)

            if esc_pressed():
                print("ESC detected, stopping image processing loop.")
                break

    if not master_rows:
        print("No detections were found; skipping CSV generation.")
        return

    df = pd.DataFrame(
        master_rows,
        columns=pd.Index(["image_path", "x1", "y1", "w", "h"])
    )
    df = df.drop_duplicates().reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    master_path = output_dir / master_name
    df.to_csv(master_path, index=False)

    train_df, val_df = split_dataframe_by_images(
        df,
        val_fraction=val_fraction,
        random_seed=seed,
    )

    train_path = output_dir / train_name
    val_path = output_dir / val_name
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    total_images = len(df["image_path"].drop_duplicates())
    print(f"Master CSV saved to: {master_path}")
    print(
        f"Split {total_images} images -> train: {len(train_df['image_path'].unique())} rows: {len(train_df)}, "
        f"val: {len(val_df['image_path'].unique())} rows: {len(val_df)}"
    )
    print(f"Train CSV saved to: {train_path}")
    print(f"Val CSV saved to: {val_path}")

def main() -> None:
    args = parse_args()
    enable_nonblocking_stdin()

    script_dir = Path(__file__).parent

    # Determine operating mode
    image_dir = _resolve_path(script_dir, args.image_dir)
    directory_mode = args.csv is None and image_dir is not None

    csv_path: Path | None = None
    if not directory_mode:
        if args.csv is None:
            default_csv = "data/splits/train.csv"
            print(f"--csv not provided; defaulting to {default_csv}")
            csv_arg = default_csv
        else:
            csv_arg = args.csv

        resolved_csv = _resolve_path(script_dir, csv_arg)
        if resolved_csv is None:
            raise ValueError("Unable to resolve CSV path")
        csv_path = resolved_csv

        if args.image_dir:
            print("Both --csv and --image-dir provided; defaulting to CSV workflow.")
    else:
        if image_dir is None:
            raise ValueError("--image-dir must be provided for directory mode")

    print("Loading RetinaFace model (serengil/retinaface)...")
    retinaface_model = RetinaFace.build_model()
    print("Press ESC at any time to stop processing early.")

    if directory_mode:
        if image_dir is None:
            raise ValueError("--image-dir must be provided for directory mode")
        resolved_output_dir = _resolve_path(script_dir, args.output_dir)
        output_dir = resolved_output_dir if resolved_output_dir is not None else (script_dir / "data/splits")
        run_directory_mode(
            image_dir=image_dir,
            retinaface_model=retinaface_model,
            threshold=args.threshold,
            allow_upscaling=args.allow_upscaling,
            output_dir=output_dir,
            master_name=args.master_name,
            train_name=args.train_name,
            val_name=args.val_name,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    else:
        resolved_data_root = _resolve_path(script_dir, args.data_root)
        data_root = resolved_data_root if resolved_data_root is not None else script_dir
        if csv_path is None:
            raise ValueError("CSV path must be resolved when running in CSV mode")
        run_csv_mode(
            csv_path=csv_path,
            data_root=data_root,
            retinaface_model=retinaface_model,
            threshold=args.threshold,
            allow_upscaling=args.allow_upscaling,
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
