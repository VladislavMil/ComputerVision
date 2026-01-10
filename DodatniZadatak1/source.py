import cv2
import numpy as np


def remove_border_objects(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    marker = np.zeros_like(binary)
    marker[0, :] = binary[0, :]
    marker[-1, :] = binary[-1, :]
    marker[:, 0] = binary[:, 0]
    marker[:, -1] = binary[:, -1]

    se = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev = np.zeros_like(marker)
    curr = marker.copy()

    while not np.array_equal(curr, prev):
        prev = curr.copy()
        curr = cv2.dilate(curr, se)
        curr = cv2.bitwise_and(curr, binary)

    result = cv2.bitwise_and(binary, cv2.bitwise_not(curr))
    return result


binary = cv2.imread("slika2.jpg", cv2.IMREAD_GRAYSCALE)
result = remove_border_objects(binary)
cv2.imwrite("result2.jpg", result)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
