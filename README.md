# Bib Tagger Python

Automatic bib number detection and metadata tagging for race photos.

Uses YOLO for bib detection and PaddleOCR for digit recognition.

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy your bib detection model
cp /path/to/best.pt models/bib_detector.pt
```

## Usage

```bash
# Process a single image
python bib_tagger.py photo.jpg

# Process a folder
python bib_tagger.py ./photos/

# With custom model and confidence
python bib_tagger.py --model models/bib_detector.pt --confidence 0.25 ./photos/

# Save debug images with bounding boxes
python bib_tagger.py --debug ./photos/

# Output results as JSON
python bib_tagger.py --json ./photos/
```

## Options

| Option | Description |
|--------|-------------|
| `--model, -m` | Path to YOLO bib detection model (default: models/bib_detector.pt) |
| `--confidence, -c` | Detection confidence threshold (default: 0.25) |
| `--box-padding` | Expand detection boxes by fraction (e.g., 0.15 = 15%) |
| `--debug` | Save annotated debug images |
| `--json` | Output results as JSON |

## How It Works

1. **Detection**: YOLO model finds bib regions in the image
2. **OCR**: PaddleOCR reads the digits from each detected bib
3. **Metadata**: Bib numbers are written to IPTC Keywords as `BIB:1234`

## Requirements

- Python 3.10+
- exiftool (for metadata writing)
