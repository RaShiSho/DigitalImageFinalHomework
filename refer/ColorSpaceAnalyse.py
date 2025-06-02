import cv2
import numpy as np
img = cv2.imread('./dog0.jpg')
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
cv2.imwrite('./RGB色彩空间/b.jpg', b)
cv2.imwrite('./RGB色彩空间/g.jpg', g)
cv2.imwrite('./RGB色彩空间/r.jpg', r)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]
cv2.imwrite("./HSV色彩空间/H.jpg",h)
cv2.imwrite("./HSV色彩空间/S.jpg", s)
cv2.imwrite("./HSV色彩空间/V.jpg", v)
