import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("coins.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, coins_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
coins_mask = cv2.morphologyEx(coins_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]

_, marker = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)
marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel, iterations=2)

recon = marker.copy()
while True:
    prev = recon.copy()
    recon = cv2.dilate(recon, kernel)
    recon = cv2.bitwise_and(recon, coins_mask)
    if np.array_equal(prev, recon):
        break

copper_mask = recon

extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=copper_mask)

plt.title("Originalna slika")
plt.imshow(img_rgb)
plt.show()

plt.title("Maska bakarnog novčića")
plt.imshow(copper_mask, cmap="gray")
plt.show()

plt.title("Izdvojeni bakarni novčić")
plt.imshow(extracted)
plt.show()
