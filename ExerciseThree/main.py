import numpy as np
import cv2 as cv

img = cv.imread('../img/MicrosoftTeams-image.png', 0)
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
print(M)

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print('x and y: ', cx, cy)
area = cv.contourArea(cnt)
print('area: ', area)
perimeter = cv.arcLength(cnt, True)
print('perimeter: ', perimeter)

x, y, w, h = cv.boundingRect(cnt)
aspect_ratio = float(w) / h
print('aspect_ratio: ', aspect_ratio)

area = cv.contourArea(cnt)
x, y, w, h = cv.boundingRect(cnt)
rect_area = w * h
extent = float(area) / rect_area
print('extent: ', extent)

area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area) / hull_area
print('solidity:', solidity)

area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4 * area / np.pi)
print('equi_diameter', equi_diameter)

(x, y), (MA, ma), angle = cv.fitEllipse(cnt)
print('Orientation:', (x, y), (MA, ma), angle)

mask = (np.random.rand(1024, 1024) > 0.5).astype(np.uint8)
cv.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))
print('mask:', mask)
print('pixelpoints', pixelpoints)

mask = np.zeros(img.shape, np.uint8)
cv.drawContours(mask, [cnt], 0, 255, -1)
pixelpoints = np.transpose(np.nonzero(mask))

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img, mask=mask)

mean_val = cv.mean(img, mask=mask)

leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])