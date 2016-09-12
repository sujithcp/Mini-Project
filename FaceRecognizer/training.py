from generals import *
import cv2
import random
import numpy as np
classifier = cv2.face.createLBPHFaceRecognizer()
def train(face_list):
    face_list = open(face_list,"r")
    faces = [i for i in face_list.read().split("\n") if i]
    random.shuffle(faces)
    images = [cv2.cvtColor(cv2.imread("res/images/OUT/"+i.split(" ")[0]),cv2.COLOR_BGR2GRAY) for i in faces]
    labels = [int(i.split(" ")[1]) for i in faces]
    print(np.array(labels))
    count = len(images)
    classifier.train(images[0:int(count/2)], np.array(labels[0:int(count/2)]))
    true,false = 0,0
    for i in range(int(count/2),len(images)):
        res = classifier.predict(images[i])
        if res[0]==labels[i]:
            true+=1
        else:
            false+=1
        print(res, labels[i],res[0]==labels[i])
    print(100*true/(true+false),"%")

train("res/images/OUT/faces_list.txt")