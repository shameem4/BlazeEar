"""
Utility script to validate images from a CSV file.
Checks if all images referenced in the CSV are valid image files.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from PIL import Image


def validate_images_from_csv(csv_path, base_path=None):
    """
    Validate all images referenced in a CSV file.

    Args:
        csv_path: Path to the CSV file containing image paths
        base_path: Optional base directory to prepend to image paths

    Returns:
        True if all images are valid, False otherwise
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    invalid_images = []
    total_images = 0
    processed_paths = set()  # Track unique image paths

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        # Check if CSV has 'image_path' column
        if not fieldnames or 'image_path' not in fieldnames:
            print(f"Error: CSV file must contain 'image_path' column")
            print(f"Found columns: {fieldnames}")
            sys.exit(1)

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            image_path = row['image_path']

            # Skip if we've already processed this image path
            if image_path in processed_paths:
                continue
            processed_paths.add(image_path)

            # Construct full path
            if base_path:
                full_path = os.path.join(base_path, image_path)
            else:
                full_path = image_path

            total_images += 1

            # Check if file exists
            if not os.path.exists(full_path):
                invalid_images.append({
                    'path': image_path,
                    'full_path': full_path,
                    'reason': 'File not found',
                    'row': row_num
                })
                continue

            # Try to open and verify as valid image
            try:
                with Image.open(full_path) as img:
                    img.verify()  # Verify it's a valid image
                # Reopen for further validation (verify() closes the file)
                with Image.open(full_path) as img:
                    img.load()  # Actually load the image data
            except Exception as e:
                invalid_images.append({
                    'path': image_path,
                    'full_path': full_path,
                    'reason': str(e),
                    'row': row_num
                })

    # Print results
    print(f"\nValidation Results:")
    print(f"Total unique images checked: {total_images}")
    print(f"Valid images: {total_images - len(invalid_images)}")
    print(f"Invalid images: {len(invalid_images)}")

    if invalid_images:
        print("\nInvalid images found:")
        print("-" * 80)
        for img in invalid_images:
            print(f"File: {img['path']}")
            print(f"Full path: {img['full_path']}")
            print(f"First occurrence: Row {img['row']}")
            print(f"Reason: {img['reason']}")
            print("-" * 80)
        return False
    else:
        print("\nAll images are valid!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate images referenced in a CSV file'
    )
    parser.add_argument(
        '--csv_file',
        default="data/splits/train.csv",
        help='Path to CSV file containing image paths'
    )
    parser.add_argument(
        '--base-path',
        help='Base directory to prepend to image paths (optional)',
        default="data/raw/blazeface"
    )

    args = parser.parse_args()

    # Validate images
    is_valid = validate_images_from_csv(args.csv_file, args.base_path)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
