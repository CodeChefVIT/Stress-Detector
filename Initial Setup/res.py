import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def make_240p():
	cap.set(3,352)
	cap.set(4,240)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
	# Capture frame by frame
	ret, frame = cap.read()
	frame1 = rescale_frame(frame, percent = 30)
	#here frame1 is rescaled to 30%
	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame', frame1) #imgshow of original frame
	cv2.imshow('gray', gray) #gray of orginal img
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
