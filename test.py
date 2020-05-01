import numpy as np
import dlib
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils

global points, points_lip, emotion_classifier

def ebdist(leye,reye):
    eyedist = dist.euclidean(leye,reye)
    points.append(int(eyedist))
    return eyedist

def lpdist(l_lower,l_upper):
    lipdist = dist.euclidean(l_lower, l_upper)
    points_lip.append(int(eyedist))
    return lipdist

def emotion_finder(faces,frame):
    dataset = ["angry", "scared", "happy", "sad", "neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = dataset[preds.argmax()]
    if label in ['scared','sad']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return label
    
def normalize_values(points,disp,points_lip,dis_lip):
    normalize_value_lip = abs(dis_lip - np.min(points_lip))/abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eye = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value = normalized_value_eye + normalize_value_lip
    strval = (np.exp(-(normalized_value)))/1.3
    print(strval)
    if strval>=100:
        return strval,"High Stress"
    else:
        return strval,"Low Stress"
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("/Users/dhairyaostwal/Desktop/Stress-Detector/Training-Code/New_Stress_Model.h5", compile=False)
cap = cv2.VideoCapture(0)
points = []
points_lip = []
while(True):
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    # lip aka mouth
    (l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth", (61,68)]


    #preprocessing the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray,0)
    for detection in detections:
        emotion = emotion_finder(detection,gray)
        cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
           
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
        openmouth = shape[l_lower:l_upper]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)
        openmouthhull = cv2.convexHull(openmouth) # figuring out convex shape when lips opened

        cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [openmouthhull], -1, (0, 255, 0), 1)
        
        # Measuring lip aka "open mouth" and eye distance
        lipdist = lpdist(l_lower[-1],l_upper[0])
        eyedist = ebdist(leyebrow[-1],reyebrow[0])

        strval = normalize_values(points,eyedist, points_lip, lipdist)
        cv2.putText(frame,"Stress Level:{}".format(str(int(strval*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

