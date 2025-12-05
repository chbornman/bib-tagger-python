# Bib Tagger

Automatic bib number detection and EXIF metadata tagging for race photography.

Feed it a folder of race photos, and it will detect bib numbers and write them to each image's IPTC Keywords metadata as `BIB:1234`. This makes photos instantly searchable by bib number in tools like Lightroom, Photo Mechanic, or any DAM software.

## Features

- **YOLO-based bib detection** - Fast, accurate detection of race bibs in photos
- **PaddleOCR digit recognition** - Reads bib numbers from detected regions
- **Smart filtering** - 50% height threshold rejects secondary numbers (gear check digits, food tickets, timing chips)
- **EXIF metadata tagging** - Writes bib numbers to IPTC Keywords via exiftool
- **Batch processing** - Process entire folders of images
- **Debug visualization** - Optional annotated output images showing detections

## Installation

### Prerequisites

- Python 3.10+
- [exiftool](https://exiftool.org/) - Required for writing metadata

```bash
# Install exiftool
# Ubuntu/Debian:
sudo apt install libimage-exiftool-perl

# macOS:
brew install exiftool

# Arch Linux:
sudo pacman -S perl-image-exiftool
```

### Setup

```bash
# Clone the repository
git clone https://github.com/chbornman/bib-tagger-python.git
cd bib-tagger-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

The pre-trained bib detection model is included at `models/bib_detector.pt`.

## Usage

### Basic Usage

```bash
# Process a single image
python bib_tagger.py photo.jpg

# Process an entire folder
python bib_tagger.py ./race_photos/
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `models/bib_detector.pt` | Path to YOLO bib detection model |
| `--confidence` | `-c` | `0.25` | Detection confidence threshold (0.0-1.0) |
| `--box-padding` | | `0.0` | Expand detection boxes by fraction (e.g., `0.15` = 15%) |
| `--debug` | | | Save annotated debug images with bounding boxes |
| `--json` | | | Output detailed results as JSON |

### Examples

```bash
# Higher confidence threshold (fewer false positives)
python bib_tagger.py --confidence 0.5 ./photos/

# Expand detection boxes by 10% (helps with tight crops)
python bib_tagger.py --box-padding 0.1 ./photos/

# Save debug images to see what was detected
python bib_tagger.py --debug ./photos/

# Get JSON output for scripting/integration
python bib_tagger.py --json ./photos/

# Combine options
python bib_tagger.py -c 0.4 --box-padding 0.15 --debug ./photos/
```

### Output

For each image processed, the tool:
1. Detects bib regions using the YOLO model
2. Reads digits from each detected region using OCR
3. Writes bib numbers to the image's IPTC Keywords as `BIB:<number>`

Example output:
```
Processing 3 image(s)...

[1/3] DSC_0001.jpg
  ✓ 1 detection(s) | Bibs: 1234 | 245ms
[2/3] DSC_0002.jpg
  ✓ 2 detection(s) | Bibs: 567, 890 | 312ms
[3/3] DSC_0003.jpg
  ✓ 0 detection(s) | 89ms

========================================
Processed: 3 images
Successful: 3
Total bibs: 3
```

When using `--debug`, annotated images are saved as `<filename>_debug.<ext>` showing:
- Green bounding boxes around detected bibs
- Detection confidence percentage
- Recognized bib number and OCR confidence

## How It Works

### Pipeline

```
Image → YOLO Detection → Box Padding → PaddleOCR → Height Filtering → Metadata Write
```

1. **Detection**: A YOLO model trained on race bib images locates bib regions
2. **Box Padding** (optional): Detection boxes are expanded to capture edge digits
3. **OCR**: PaddleOCR extracts text from each detected region
4. **Digit Filtering**: Non-digit characters are removed
5. **Height Filtering**: Text regions smaller than 50% of the tallest region are rejected
6. **Selection**: The largest remaining text region is selected as the bib number
7. **Metadata**: Bib numbers are written to IPTC Keywords using exiftool

### Why 50% Height Filtering?

Race bibs often contain multiple numbers:
- **Main bib number** (large, prominent)
- Gear check numbers (small)
- Food/drink ticket numbers (small)
- Timing chip codes (small)
- Wave/corral indicators (small)

The 50% height threshold keeps only text regions that are at least half as tall as the largest detected text. This reliably filters out secondary numbers while keeping the main bib number, even when multiple runners appear in frame.

### Why Area-Based Selection?

After height filtering, if multiple candidates remain (e.g., two runners with similar-sized bibs), the tool selects the one with the largest bounding box area. This tends to favor the runner closest to the camera—usually the intended subject.

## Metadata Format

Bib numbers are written to IPTC Keywords with the format:
```
BIB:1234
```

This format:
- Is searchable in Lightroom, Photo Mechanic, and most DAM software
- Doesn't conflict with other keywords you might use
- Makes it easy to search for specific runners: `BIB:1234`
- Allows wildcard searches: `BIB:*` (all tagged photos)

To verify metadata was written:
```bash
exiftool -Keywords photo.jpg
```

## Supported Formats

Input images:
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- TIFF (`.tif`, `.tiff`)

The tool skips files with `_debug` in the filename to avoid reprocessing debug output.

## Troubleshooting

### "No bibs detected"
- Try lowering the confidence threshold: `--confidence 0.15`
- Check if bibs are clearly visible and not too small in frame
- Use `--debug` to see what regions are being detected

### Wrong numbers detected
- Try increasing confidence: `--confidence 0.5`
- Use `--box-padding 0.1` if digits are being cut off at edges
- The 50% height filter should reject most secondary numbers automatically

### "Failed to write metadata"
- Ensure exiftool is installed: `exiftool -ver`
- Check file permissions on the image
- Verify the image format supports IPTC metadata

### Slow processing
- Processing runs on CPU by default
- First image is slower due to model loading
- Typical speed: 200-400ms per image on modern hardware

## Dependencies

- [ultralytics](https://github.com/ultralytics/ultralytics) - YOLO implementation
- [paddleocr](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [opencv-python](https://opencv.org/) - Image processing
- [exiftool](https://exiftool.org/) - Metadata writing (system dependency)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- YOLO model architecture by [Ultralytics](https://ultralytics.com/)
- OCR powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Inspired by the C++ [SonyTagger](https://github.com/chbornman/SonyTagger) project
