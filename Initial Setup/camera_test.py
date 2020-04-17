import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
	# Capture frame by frame
	ret, frame = cap.read()

	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame', frame) #imgshow of original frame
	cv2.imshow('gray', gray) #gray of orginal img
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
