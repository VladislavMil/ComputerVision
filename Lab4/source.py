import cv2
import numpy as np
import matplotlib.pyplot as plt

big_image = cv2.imread("./canvas.png")
big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB)

# 288:1007, 385:1824
cropped_image = big_image[285:1005, 385:1820]
net = cv2.dnn.readNetFromCaffe(
    "./GoogLeNet/bvlc_googlenet.prototxt", "./GoogLeNet/bvlc_googlenet.caffemodel"
)
with open("./GoogLeNet/classification_classes_ILSVRC2012.txt", "r") as f:
    class_labels = f.read().strip().split("\n")
sizes = [((180, 180), 180), ((360, 360), 360), ((720, 720), 720)]
confidence_threshold = 90
margin = 3
for window_size, step_size in sizes:
    for y in range(0, cropped_image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, cropped_image.shape[1] - window_size[0] + 1, step_size):
            roi = cropped_image[y : y + window_size[1], x : x + window_size[0]]
            blob = cv2.dnn.blobFromImage(roi, 1, (224, 224), (104, 117, 123))
            net.setInput(blob)
            preds = net.forward()
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx] * 100
            if confidence >= confidence_threshold:
                class_label = class_labels[class_idx]
                print(class_label, confidence)
                if "dog" in class_label:
                    cv2.rectangle(
                        cropped_image,
                        (x + margin, y + margin),
                        (x + window_size[0] - margin, y + window_size[1] - margin),
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        cropped_image,
                        "DOG",
                        (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2,
                    )
                elif "cat" in class_label:
                    cv2.rectangle(
                        cropped_image,
                        (x + margin, y + margin),
                        (x + window_size[0] - margin, y + window_size[1] - margin),
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        cropped_image,
                        "CAT",
                        (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
cv2.imwrite("output.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
plt.imshow(cropped_image)
plt.axis("off")
plt.show()
