#
# @Author: Thejus Singh Jagadish
# @Date Created: 6th Feb 2017
#

from scipy.spatial import distance as d
import numpy as np 
import cv2
import mahotas
import imutils


def describe_shape(image):
	features = []
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (13,13), 0)
	threshold = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[1]

	#Closing operation
	threshold = cv2.dilate(threshold, None, iterations=4)
	threshold = cv2.erode(threshold, None, iterations=2)

	contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contour = contour[0] if imutils.is_cv2() else contour[1]

	for c in contour:
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		(x, y, w, h) = cv2.boundingRect(c)
		roi = mask[y:y+h, x:x+w]

		feature = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
		features.append(feature)

	return (contour, features)


refImg = cv2.imread("pokemon_red.png")
(_, refFeature) = describe_shape(refImg)

img = cv2.imread("shapes.png")
(contour, features) = describe_shape(img)

D = d.cdist(refFeature, features)
i = np.argmin(D)

for (j, c) in enumerate(contour):

	if i != j:
		box = cv2.minAreaRect(c)
		box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2 else cv2.BoxPoints(box))
		cv2.drawContours(img, [box], -1, (0, 0, 255), 2)

box = cv2.minAreaRect(contour[i])
box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2 else cv2.BoxPoints(box))
cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
(x, y, w, h) = cv2.boundingRect(contour[i])
cv2.putText(img, "Found!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

cv2.imshow("Image", img)
cv2.imshow("Input Image", refImg)
cv2.waitKey(0)