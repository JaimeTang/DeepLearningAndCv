import cv2
import numpy as np

img = cv2.imread('img\img.jpg')
print(img)
cv2.imshow('img',img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()