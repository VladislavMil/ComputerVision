import numpy as np
import cv2 as cv

slika1 = cv.imread('1.jpg')
slika2 = cv.imread('2.jpg')
slika3 = cv.imread('3.jpg')


def panorama_od_tri_slike():
    img = panorama_od_dve_slike(slika2, slika3)
    img = panorama_od_dve_slike(slika1, img)
    return img

def panorama_od_dve_slike(imgL, imgR):
    detector = cv.SIFT_create()

    kp1, des1 = detector.detectAndCompute(imgR, None)
    kp2, des2 = detector.detectAndCompute(imgL, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)


    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 30:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 30))
        return None

    width = imgL.shape[1] + imgR.shape[1]
    height = imgL.shape[0] + int(imgR.shape[0] / 2)
    outimg = cv.warpPerspective(imgR, M, (width, height))
    outimg[0:imgL.shape[0], 0:imgL.shape[1]] = imgL

    outimg = trim(outimg)
    return outimg

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame

panorama = panorama_od_tri_slike()

cv.imshow("Panorama", panorama)
cv.imwrite("panorama.jpg", panorama)
cv.waitKey(0)