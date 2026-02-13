import cv2
from ultralytics import YOLO

def main():
    # Load a standard YOLOv8 model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # COCO Class IDs for drink-related items
    # 39: bottle, 41: cup, 40: wine glass, 45: bowl (sometimes used for soup/drinks)
    DRINK_CLASS_IDS = [39, 40, 41, 45]

    # Open the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Drink Detection.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, classes=DRINK_CLASS_IDS, conf=0.25, verbose=False)

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence and class name
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label with "Potential Drink"
                label = f"Potential Drink ({cls_name}: {conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Stage 1: Container Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
