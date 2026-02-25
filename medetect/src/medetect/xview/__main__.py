import argparse
from pathlib import Path
from medetect.xview.convert import convert_xview_to_yolo

def main():
    parser = argparse.ArgumentParser(description="Convert xView dataset to YOLO format.")
    parser.add_argument("--geojson_path", type=Path, required=True, help="Path to xView geojson file.")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory containing xView images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save YOLO labels.")
    args = parser.parse_args()

    convert_xview_to_yolo(
        geojson_path=args.geojson_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
