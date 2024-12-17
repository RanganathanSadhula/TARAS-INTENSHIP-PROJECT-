import cv2
import numpy as np

# Load the YOLOv7 model (you need to have the correct weight and config files for YOLOv7)
net = cv2.dnn.readNet('yolov7.weights', 'yolov7.cfg')

# List of classes (ensure this list corresponds to the trained model's classes)
classes = ["Ambulance"]

# Set up the video capture
cap = cv2.VideoCapture("1.mp4")

# Color for bounding boxes
colors = np.random.uniform(0, 255, size=(100, 3))

# Define the font for text
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, img = cap.read()
    if img is None:
        break  # End of video
    
    # Resize the image (optional)
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(img, 1/255, (640, 640), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get the output layers
    output_layers = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers)

    # Initialize lists for class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Loop through the detections
    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold (can be adjusted)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Store the information
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                print(f"Detected object confidence: {confidence:.2f}")

    # Apply NMS (Non-Maximum Suppression)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Indexes: {indexes}")

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Get the class label
            color = colors[class_ids[i]]  # Set the color for the bounding box

            # Draw the rectangle and label on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 2, (0, 255, 0), 2)

    # Display the output image
    cv2.imshow("YOLOv7 Object Detection", img)

    # Press 'Esc' to exit
    key = cv2.waitKey(27)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
