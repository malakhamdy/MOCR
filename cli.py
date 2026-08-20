"""Small CLI wrapper around the extraction pipeline.

Usage:
    python cli.py path/to/card.jpg

The CLI prints the structured JSON result. It does not save uploaded images.
"""
import argparse
import json
import cv2
from core.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Egyptian ID OCR pipeline")
    parser.add_argument("image", help="path to an ID image")
    parser.add_argument("--no-ocr", action="store_true", help="run detection/localization only")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    result = run_pipeline(image, run_ocr=not args.no_ocr)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
