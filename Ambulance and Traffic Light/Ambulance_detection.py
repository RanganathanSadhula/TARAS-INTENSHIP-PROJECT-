import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last_2.weights', 'yolov3_testing.cfg')

classes = ["Ambulance"]


cap = cv2.VideoCapture("1.mp4")
font = cv2.QT_FONT_BOLD
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    class_ids = []
    confidences = []
    boxes = []
    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                print(f"Detected object confidence: {confidence:.2f}")

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (100, 200, 100), 2)
            cv2.putText(img, label, (x, y + 30), font, 3, (0,255,0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(27)

cap.release()
cv2.destroyAllWindows()
