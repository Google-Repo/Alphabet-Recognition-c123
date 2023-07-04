#importing modules here:
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import  fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

##Load and read the 'image.npz' file:
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts)
classes =  [["A","B","C","D","E","F",'G','H',"J","K","L","M","N","O","P","Q","R","S","T"]]
nclasses = len(classes)

##spliting the data into train and test mode:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 20, random_state= 40)

#scaling the feature:
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model:
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled,y_train)

##Calculating the accuracy of the model:
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is:-", accuracy)

#Starting the camera:
cap = cv2.VideoCapture(0)

while(True):
##Code to open the camera and start reading the frames:
    #Capture frame-by-frame
    try:
        ret,frame = cap.read()
        
        #Our operations on the frame come here:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
    ##Draw a box at the center of the screen. 
    ##And consider only the area inside the box to detect the images.
        height, width =gray.shape
        upper_left = (int(width/2-56), int(height/ 2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right,(0,255,0),2)
        
        roi = gray[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
        
        ##Converting cv2 image to pil(pillow) format:
        im_pil = Image.fromarray(roi)
        
        ## Convert to grayscale image -'l', format means each pixel is
        ## represented by single value from 0 to 255:
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        ##Using clip function to limit the values between 0,255
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - min_pixel, 0,255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        ##Create a test sample and make a prediction
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ",test_pred)
        
        ##Display the resulting frame:
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
    
##When everything done, release the capture:
cap.release()
cv2.destroyAllWindows()
        
        