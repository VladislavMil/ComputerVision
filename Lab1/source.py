import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('slika.png', 0)
plt.imshow(img, cmap='gray')
plt.title('Originalna slika')
plt.show()


f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

magnitude_spectrum = np.log(np.abs(fshift))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra pre uklanjanja šuma')
plt.savefig('fft_mag.png')
plt.show()

x1, y1 = 231, 231
x2, y2 = 281, 281
x3, y3 = 356, 156
x4, y4 = 156, 356

fshift[x1 , y1] = 0
fshift[x2 , y2] = 0
fshift[x3 , y3] = 0
fshift[x4 , y4] = 0

# Rešenje za log(0), offset
epsilon = 1e-10 

# Računanje sa ofsetom
magnitude_spectrum = np.log(np.abs(fshift) + epsilon)

#magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja šuma')
plt.savefig('fft_mag_filtered.png')
plt.show()

f_ishift = np.fft.ifftshift(fshift)

img_filtered = np.fft.ifft2(f_ishift).real

plt.imshow(img_filtered, cmap='gray')
plt.title('Sređena slika')
plt.show()
cv2.imwrite('output.png', img_filtered)