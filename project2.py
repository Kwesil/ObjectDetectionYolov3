#  Using Yolov3 to detect objects
import cv2
import numpy as np

# load yolo
modelConfig = r"C:\Users\Nwabike\Desktop\Tech\opencv\Yolo project\yolov3.cfg"
modelWeights = r"C:\Users\NWabike\Desktop\Tech\opencv\Yolo project\yolov3.weights"
net = cv2.dnn.readNet(modelConfig, modelWeights)

#  dnn is a neural network

classes = []

# in classes you can specify objects that you want to detect
# classes = ['car', 'person', 'bicycle']
names = r"C:\Users\Nwabike\Desktop\Tech\opencv\Yolo project\coco.names"

with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
print(classes)
print(len(classes))

# loading classes from coco file
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#  with output layer we get detection of objects
colors = np.random.uniform(1, 255, size=(len(classes), 3))

# Loading live cam
video = cv2.VideoCapture(0)
 
while True:
    # reading the live cam
    ret, img = video.read()
    height, width, channels = img.shape

    #  Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # True refers to converting into rgb format since opencv uses bgr.

    net.setInput(blob)

    # passing blob into yolo algorithm
    outs = net.forward(output_layers)
    # Giving network to output layers for final result

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # NMS - non max suppression
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[1]
            label = str(classes[class_ids[1]])
            color = colors[1]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            # draw rectangle around boxes."2" is the width of the bar
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            # text in box to label the object
        
    cv2.imshow("image", img)
    if cv2.waitKey(10) == ord("q"):
        break
    # waitkey stops the output

video.release()
cv2.destroyAllWindows()
print("Code Completed")
