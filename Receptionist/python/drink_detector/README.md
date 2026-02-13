# Drink Detector for HRI Challenge

## IMPORTANT NOTE
This was used for the old spec project but is no longer being used. Can be used for other projects where drink detection is needed.

This project uses **YOLO-World** (Real-time Open-Vocabulary Object Detection) to identify drinks. 

## Why YOLO-World?
The competition rules state: *The favorite drinks of the guests can be any english named popular drink (it may not be in the list of objects provided).*

Standard YOLO models (trained on COCO) only detect generic classes like "bottle" or "cup". Training a custom model requires a dataset of every possible drink, which is impossible if the drinks are unknown until the event.

YOLO-World allows us to define classes via text prompts at runtime. We can simply list "orange juice", "coke", "water", etc., and the model attempts to find objects matching those descriptions without any extra training.

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the detection script:
```bash
python detect_drinks.py
```

## Configuration
Edit the `drink_classes` list in `detect_drinks.py` to add or remove drink types you expect to see.

```python
drink_classes = [
    "bottle of water",
    "orange juice", 
    "coke can",
    # ... add more here
]
```
