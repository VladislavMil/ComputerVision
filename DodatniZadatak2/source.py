import cv2
import numpy as np

img = cv2.imread("ulaz.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ref_contour = None
max_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if area > max_area and x < img.shape[1] // 2 and y < img.shape[0] // 2:
        max_area = area
        ref_contour = cnt

output = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200:
        continue
    similarity = cv2.matchShapes(ref_contour, cnt, cv2.CONTOURS_MATCH_I1, 0)

    if similarity < 0.15:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite("izlaz.png", output)

cv2.imshow("Izlaz", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
