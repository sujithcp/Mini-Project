import cv2
import os
import re

face_cascade = cv2.CascadeClassifier("res/haarcascade_profileface.xml")


def get_clean_path(path):
    """
    replace extra '/' in url
    """
    return re.sub("//*", "/", path)


def get_faces_from_image(image):
    """
    faces:array
    for each face in image
        append face to faces
    return faces
    """
    #window = cv2.namedWindow("Image")
    faces = []
    rects = face_cascade.detectMultiScale(image)
    for (x, y, w, h) in rects:
        face = image[y:y+h, x:x+w]
        if face.shape[0]<250 or face.shape[1]<250:
            print("Skipping: Too small size")
            continue
        faces.append(face)
        #cv2.imshow(window, face)
        #cv2.waitKey(2)
    return faces


def extract_faces_from_dir(dir):  # extract faces from subdirectories of dir
    """
    map subdirs of dir to integer in file class_map
    for each subdir in dir
        for each file in subdir (not subdir/OUT)
            extract faces from subdir to dir/OUT folder
            add face and class to faces_list.txt file
    """
    class_map = {}
    OUT_PATH = dir + "/OUT"
    print(dir)
    if not os.path.isdir(dir):  # check if directory exists
        return
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)
    face_list = open(OUT_PATH + "/faces_list.txt", "w+")
    class_map = {j:i for i,j in enumerate(os.listdir(dir)) if j != "OUT"}
    map = open(OUT_PATH + "/class_map.txt", "w")
    for i in class_map:
        print(i,class_map[i])
        map.write(i + " " + str(class_map[i]) + "\n")
    print(class_map)
    for cls in class_map:
        print(cls)
        if not os.path.isdir(get_clean_path(dir + "/" + cls)):  # check if directory exists
            continue
        count = 0
        for f in os.listdir(get_clean_path(dir + "/" + cls)):
            print(f)
            if not os.path.isfile(get_clean_path(dir + "/" + cls + "/" + f)):  # check if item a file
                continue
            faces = get_faces_from_image(cv2.imread(dir + "/" + cls + "/" + f))
            for face in faces:
                cv2.imwrite(OUT_PATH + "/" + cls + "-" + str(count)+".jpg", face)
                face_list.write(cls + "-" + str(count)+".jpg "+str(class_map[cls])+"\n")
                count += 1

