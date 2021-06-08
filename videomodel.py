import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



def Analyzer(img_array):
    image_ = cv2.resize(img_array,(150,150))
    image_ = np.expand_dims(image_,axis=0)
    image_ = image_/255.0
    prediction = model.predict(image_)
    return prediction 

def Face_Detector(face_haarcascade,eye_haarcascade):

    cap = cv2.VideoCapture(0)
    face = cv2.CascadeClassifier(face_haarcascade)
    eyes  =cv2.CascadeClassifier(eye_haarcascade)


    while True:
        ret,frame = cap.read()
        frame = imutils.resize(frame,width=500)


        faces = face.detectMultiScale(frame)
        

        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,250,0),2)
            cv2.rectangle(frame,(1,1),(180,50),(0,250,0),-1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame,'SAFE',(50,35), font, 1, (0,0,0))
            

            sub_frame = frame[y:y+h,x:x+w]

            eye = eyes.detectMultiScale(sub_frame)
            for xe,ye,we,he in eye:
                cv2.rectangle(sub_frame,(xe,ye),(xe+we,ye+he),(0,250,0),2)
                

                img_data = sub_frame[ye:ye+he,xe:xe+we]
                
                try:
                    result = Analyzer(img_data)
                    if result>0.5:
                        print("open")
                    else:
                        cv2.rectangle(frame,(1,1),(180,50),(0,0,250),-1)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,250),2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame,'DANGER!!',(25,35), font, 1, (0,0,0))
                        print("close")
                except Exception as e:
                    print(e)
                
                


        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    
    model_path = r"C:\virtual enviroments\tfod-project\dd-project\models\models\dd-model1.h5"
    model = load_model(model_path)
    Face_Detector(r"C:\virtual enviroments\tfod-project\dd-project\models\haarcascade files\haarcascade_frontalface_default.xml",r"C:\virtual enviroments\tfod-project\dd-project\models\haarcascade files\haarcascade_eye_tree_eyeglasses.xml")






    
