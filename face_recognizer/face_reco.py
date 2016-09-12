#!/usr/bin/python3

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import re

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training

    subjects = [i for i in os.listdir(path) if not i=="TEST"]
    image_paths=[]
    for sub in subjects:
    	image_paths.extend([os.path.join(os.path.join(path,sub), f) for f in os.listdir(path+"/"+sub) if not re.search(".*ANS[.]JPG",f)])
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels=[]
    print(labels)
    for image_path in image_paths:
        # Read the image and convert to grayscale
        print(image_path)
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for i,(x, y, w, h) in enumerate(faces):
        	im = cv2.resize(image[y: y + h, x: x + w],(250,250))
        	images.append(im)
        	out_path = "./OUT/"+image_path[2:]+str(i)+".jpg"
        	if(re.search("JITH",image_path)):
        		print(out_path)
        		cv2.imwrite(out_path,im)
        	labels.append(int(image_path[13:15]))
        	cv2.imshow("Adding faces to traning set...", im)
        	cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './OUT/data'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()
print(labels)
# Perform the tranining
recognizer.train(images, np.array(labels))
path="./data"
print("SUCKS")
# Append the images with the extension .sad into image_paths
image_paths=[]
image_paths.extend([os.path.join(os.path.join(path,"TEST"), f) for f in os.listdir(path+"/TEST/")])
print(image_paths)
for image_path in image_paths:
    print(image_path)
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
    	im = cv2.resize(predict_image[y: y + h, x: x + w],(350,350))
    	if(im.shape[:2][0]<100):
    		continue
    	nbr_predicted, conf = recognizer.predict(im)
    	nbr_actual = int(image_path[12:14])
    	if nbr_actual == nbr_predicted:
    		print("{} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
    	else:
    		print("{} is Incorrect Recognized as {} conf = {}".format(nbr_actual, nbr_predicted,conf))
    	cv2.imshow("Recognizing Face", im)
    	cv2.waitKey(1000)
