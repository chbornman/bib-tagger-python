#!/usr/bin/env python3
"""
Bib Tagger - Automatic bib number detection and metadata tagging.

Uses YOLO for bib detection and PaddleOCR for digit recognition.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

# Suppress PaddleOCR verbose output
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
os.environ['GLOG_minloglevel'] = '3'

from paddleocr import PaddleOCR


class BibTagger:
    """Detects bib numbers in images and writes them to metadata."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        ocr_lang: str = 'en',
        box_padding: float = 0.0,
        debug: bool = False,
    ):
        self.confidence = confidence
        self.box_padding = box_padding
        self.debug = debug

        # Load YOLO model for bib detection
        print(f"Loading bib detector: {model_path}")
        self.detector = YOLO(model_path)

        # Initialize PaddleOCR for digit recognition
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            lang=ocr_lang,
            use_textline_orientation=True,
        )
        print("Ready!")

    def detect_bibs(self, image: cv2.Mat) -> list[dict]:
        """Detect bib regions in an image."""
        results = self.detector(image, conf=self.confidence, verbose=False, device='cpu')

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Apply padding if configured
                if self.box_padding > 0:
                    w, h = x2 - x1, y2 - y1
                    pad_x = int(w * self.box_padding)
                    pad_y = int(h * self.box_padding)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(image.shape[1], x2 + pad_x)
                    y2 = min(image.shape[0], y2 + pad_y)

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': conf,
                })

        return detections

    def read_bib_number(self, image: cv2.Mat, box: tuple) -> Optional[tuple[str, float]]:
        """Read the bib number from a detected region using OCR."""
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Run PaddleOCR
        result = self.ocr.predict(crop)

        if not result:
            return None

        # Extract text regions with bounding boxes
        texts = []
        confidences = []
        boxes = []

        for page_result in result:
            if 'rec_texts' in page_result and 'rec_scores' in page_result:
                for i, text in enumerate(page_result.get('rec_texts', [])):
                    conf = page_result.get('rec_scores', [0.0])[i] if i < len(page_result.get('rec_scores', [])) else 0.0

                    # Filter to only digits
                    digits_only = ''.join(c for c in str(text) if c.isdigit())

                    if digits_only:
                        texts.append(digits_only)
                        confidences.append(float(conf))

                        if 'dt_polys' in page_result and i < len(page_result['dt_polys']):
                            poly = page_result['dt_polys'][i]
                            xs = [float(p[0]) for p in poly]
                            ys = [float(p[1]) for p in poly]
                            boxes.append([min(xs), min(ys), max(xs), max(ys)])
                        else:
                            boxes.append(None)

        if not texts:
            return None

        # Apply 50% height threshold filtering to reject secondary numbers
        # (gear check digits, food tickets, etc. that appear smaller on bibs)
        if boxes and any(b is not None for b in boxes):
            heights = [(b[3] - b[1]) if b else 0 for b in boxes]
            max_height = max(heights) if heights else 0

            if max_height > 0:
                # Keep only text regions with height >= 50% of max
                filtered = [
                    (texts[i], confidences[i], boxes[i])
                    for i in range(len(texts))
                    if boxes[i] is not None and (boxes[i][3] - boxes[i][1]) >= max_height * 0.5
                ]

                if filtered:
                    texts = [f[0] for f in filtered]
                    confidences = [f[1] for f in filtered]
                    boxes = [f[2] for f in filtered]

        # Return largest text region (likely main bib number)
        if boxes and any(b is not None for b in boxes):
            areas = [(b[2] - b[0]) * (b[3] - b[1]) if b else 0 for b in boxes]
            max_idx = areas.index(max(areas))
            return texts[max_idx], confidences[max_idx]

        return texts[0], confidences[0] if confidences else 0.0

    def write_metadata(self, image_path: str, bib_numbers: list[str]) -> bool:
        """Write bib numbers to image metadata using exiftool."""
        if not bib_numbers:
            return True

        # Build IPTC keywords with BIB: prefix
        keywords = [f"BIB:{bib}" for bib in bib_numbers]
        keyword_args = ' '.join(f'-Keywords+="{kw}"' for kw in keywords)

        cmd = f'exiftool -overwrite_original {keyword_args} "{image_path}"'
        result = os.system(cmd + ' > /dev/null 2>&1')

        return result == 0

    def process_image(self, image_path: str) -> dict:
        """Process a single image: detect bibs, read numbers, write metadata."""
        result = {
            'path': image_path,
            'success': False,
            'bibs': [],
            'detections': 0,
            'time_ms': 0,
            'error': None,
        }

        start_time = time.time()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            result['error'] = 'Failed to load image'
            return result

        # Detect bibs
        detections = self.detect_bibs(image)
        result['detections'] = len(detections)

        # Read bib numbers from each detection
        for det in detections:
            ocr_result = self.read_bib_number(image, det['box'])
            if ocr_result:
                bib_number, confidence = ocr_result
                result['bibs'].append({
                    'number': bib_number,
                    'confidence': confidence,
                    'box': det['box'],
                })

        # Write metadata if bibs found
        if result['bibs']:
            bib_numbers = [b['number'] for b in result['bibs']]
            if self.write_metadata(image_path, bib_numbers):
                result['success'] = True
            else:
                result['error'] = 'Failed to write metadata'
        else:
            result['success'] = True  # No bibs is not an error

        # Save debug image if requested
        if self.debug and detections:
            self._save_debug_image(image, image_path, detections, result['bibs'])

        result['time_ms'] = (time.time() - start_time) * 1000
        return result

    def _save_debug_image(self, image: cv2.Mat, image_path: str, detections: list, bibs: list):
        """Save annotated debug image."""
        debug_img = image.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']

            # Draw bounding box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Build label
            label = f"{det['confidence']*100:.1f}%"
            if i < len(bibs):
                label += f" -> {bibs[i]['number']} ({bibs[i]['confidence']*100:.1f}%)"

            # Draw label
            cv2.putText(debug_img, label, (x1, y1 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        # Save
        path = Path(image_path)
        debug_path = path.parent / f"{path.stem}_debug{path.suffix}"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"  Debug: {debug_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Bib Tagger - Automatic bib number detection and metadata tagging'
    )
    parser.add_argument(
        'input',
        help='Image file or folder to process'
    )
    parser.add_argument(
        '--model', '-m',
        default='models/bib_detector.pt',
        help='Path to YOLO bib detection model (default: models/bib_detector.pt)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.25,
        help='Detection confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--box-padding',
        type=float,
        default=0.0,
        help='Expand detection boxes by fraction (default: 0, e.g., 0.15 = 15%%)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save annotated debug images'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Initialize tagger
    tagger = BibTagger(
        model_path=str(model_path),
        confidence=args.confidence,
        box_padding=args.box_padding,
        debug=args.debug,
    )

    # Collect images to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    if input_path.is_dir():
        images = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
            and '_debug' not in f.stem
        ])
    else:
        images = [input_path]

    if not images:
        print("No images found")
        sys.exit(1)

    print(f"\nProcessing {len(images)} image(s)...\n")

    # Process images
    results = []
    total_bibs = 0
    success_count = 0

    for i, img_path in enumerate(images, 1):
        if len(images) > 1:
            print(f"[{i}/{len(images)}] {img_path.name}")

        result = tagger.process_image(str(img_path))
        results.append(result)

        # Print result
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['detections']} detection(s)", end='')

        if result['bibs']:
            bibs_str = ', '.join(b['number'] for b in result['bibs'])
            print(f" | Bibs: {bibs_str}", end='')
            total_bibs += len(result['bibs'])

        print(f" | {result['time_ms']:.0f}ms")

        if result['error']:
            print(f"  Error: {result['error']}")

        if result['success']:
            success_count += 1

    # Summary
    print(f"\n{'='*40}")
    print(f"Processed: {len(images)} images")
    print(f"Successful: {success_count}")
    print(f"Total bibs: {total_bibs}")

    # JSON output
    if args.json:
        print("\n" + json.dumps(results, indent=2))

    sys.exit(0 if success_count > 0 else 1)


if __name__ == '__main__':
    main()
