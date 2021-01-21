import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/AI/Pictures/Saved Pictures20201229_200001.jpg', 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('A', cv.Canny(gray, 200, 300))
cv.waitKey()
cv.destroyAllWindows()
