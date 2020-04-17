import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/Users/dhairyaostwal/Desktop/haarcascades/haarcascade_frontalface_alt2.xml')



cap = cv2.VideoCapture(0)
# ...
#lowering resolution
def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_480p()



while True:
	# Capture frame by frame
	ret, frame = cap.read()

	gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)
	for(x,y,w,h) in faces:
		print(x,y,w,h)

		#recognise? deep learned model prediction keras tensorflow pytorch scikit learn




		color = (255,0,0) #BGR 0-255
		stroke = 2
		en_cord_x = x + w
		en_cord_y = y + h
		cv2.rectangle(frame, (x,y), (en_cord_x, en_cord_y), color, stroke)  



	#Display the resulting frame
	cv2.imshow('frame', frame) #imgshow of original frame
	#cv2.imshow('gray', gray) #gray of orginal img
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
