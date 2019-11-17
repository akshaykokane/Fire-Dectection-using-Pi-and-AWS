import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time
from send_sms import send_sms
from save_to_s3 import upload_to_aws


#Establish  MQTT Server
import paho.mqtt.publish as publish
MQTT_SERVER = "192.168.1.31"
MQTT_PATH = "test_channel"

#publish.single(MQTT_PATH, "Hello World!", hostname=MQTT_SERVER)
# loading the stored model from file
print('LOading MOdel')
model = load_model(r'Fire-64x64-color-v7-soft.h5')
print('Model loaded')

#cap = cv2.VideoCapture(r'video1.mp4')
cap = cv2.VideoCapture(0)
time.sleep(12)

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False


IMG_SIZE = 64
# IMG_SIZE = 224


#for i in range(2500):
#    cap.read()

imgCount = 0;

while(1):
    imgCount = imgCount + 1
    rval, image = cap.read()
    if rval==True:
        orig = image.copy()
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        tic = time.time()
        fire_prob = model.predict(image)[0][0] * 100
        toc = time.time()
        print("Time taken = ", toc - tic)
        print("FPS: ", 1 / np.float64(toc - tic))
        print("Fire Probability: ", fire_prob)
        print("Predictions: ", model.predict(image))
        print(image.shape)
        
        lower = [18, 50, 50]
        higher = [45, 255, 255]
        color_map_frame = cv2.applyColorMap(orig, cv2.COLORMAP_JET)  
        
        #lower = np.array(lower, dtype="uint8")
        #upper = np.array(higher, dtype="uint8")
           
        #blur = cv2.GaussianBlur(color_map_frame, (21, 21), 0)
        #hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
           
        #mask = cv2.inRange(blur, lower, upper)
           
        #output1 = cv2.bitwise_and(color_map_frame, blur, mask=mask)
        
        cv2.imshow("output1", color_map_frame)
        key1= cv2.waitKey(10)
        if key1 == 27: # exit on ESC
           break    
        if fire_prob > 99:
           print("..............Fire Detected .. ")
           #Give the filename
           
           img_name = "{}.png".format(imgCount)
           #save image file with the above file name
           cv2.imwrite(img_name, orig)
           
           
           #upload the frame where fire was detected
           upload_to_aws(img_name,'fire-detection-ads', img_name)
           #send sms
           link = send_sms("FIRE DETECTED ", img_name);
           publish.single(MQTT_PATH, link, hostname=MQTT_SERVER)
           time.sleep(100)
           
        label = "Fire Probability: " + str(fire_prob)
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

        cv2.imshow("Output", orig)
        
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
    elif rval==False:
            break
end = time.time()


cap.release()
cv2.destroyAllWindows()
