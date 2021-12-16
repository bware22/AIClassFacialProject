import cv2
import numpy as np
import dlib
import os
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

leftEye = 0
rightEye = 0
upper = 0
lower = 0
data = []
picFile = []
point = []
count = 0

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def eyeDistAll():
    counter = 0
    for filename in os.listdir("Cropped"):
        if filename.endswith(".jpg"):
            print(counter)
            counter = counter + 1
            imgName = os.path.join("Cropped", filename)
            img = cv2.imread(imgName)
            gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                # x1 = face.left() # left point
                # y1 = face.top() # top point
                # x2 = face.right() # right point
                # y2 = face.bottom() # bottom point

                # Create landmark object
                landmarks = predictor(image=gray, box=face)
            x = landmarks.part(39).x
            y = landmarks.part(39).y
            leftEye = (x, y)
            x = landmarks.part(42).x
            y = landmarks.part(42).y
            rightEye = (x, y)

            distance = [math.sqrt(((rightEye[0] - leftEye[0])**2) + ((rightEye[1] - leftEye[1])**2))]
            picFile.append(imgName)
            data.append(distance)

        else:
            continue

    data2 = np.array(data)
    kmeans = KMeans(n_clusters=4).fit(data2)
    getEyeGroups(data)
    plt.scatter(data2[:, 0], data2[:, 0], c=kmeans.labels_, cmap='rainbow')
    plt.title('Distance between eyes')
    plt.show()
    print(kmeans.cluster_centers_)


def mouthDistAll():
    counter = 0
    for filename in os.listdir("Cropped"):
        if filename.endswith(".jpg"):
            print(counter)
            counter = counter + 1
            imgName = os.path.join("Cropped", filename)
            img = cv2.imread(imgName)
            gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                # x1 = face.left() # left point
                # y1 = face.top() # top point
                # x2 = face.right() # right point
                # y2 = face.bottom() # bottom point

                # Create landmark object
                landmarks = predictor(image=gray, box=face)
            x = landmarks.part(48).x
            y = landmarks.part(48).y
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            leftEye = (x, y)
            x = landmarks.part(54).x
            y = landmarks.part(54).y
            rightEye = (x, y)
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

            distance = [math.sqrt(((rightEye[0] - leftEye[0])**2) + ((rightEye[1] - leftEye[1])**2))]
            picFile.append(imgName)
            data.append(distance)

        else:
            continue

    data2 = np.array(data)
    kmeans = KMeans(n_clusters=4).fit(data2)
    getMouthGroups(data)
    plt.scatter(data2[:, 0], data2[:, 0], c=kmeans.labels_, cmap='rainbow')
    plt.title('Length of mouth')
    plt.show()
    print(kmeans.cluster_centers_)


def eyeDistSpecific(imageName):
    img = cv2.imread(imageName)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)
    x = landmarks.part(39).x
    y = landmarks.part(39).y
    cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    leftEye = (x, y)
    x = landmarks.part(42).x
    y = landmarks.part(42).y
    cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    rightEye = (x, y)
    print(leftEye, rightEye)
    distance = [math.sqrt(((rightEye[0] - leftEye[0]) ** 2) + ((rightEye[1] - leftEye[1]) ** 2))]
    data.append(distance)
    data2 = np.array(data)
    kmeans = KMeans(n_clusters=4)
    plt.scatter(data2[:, 0], data2[:, 0], cmap='rainbow')
    plt.title('Distance between eyes')
    plt.show()
    cv2.imshow(winname="Face", mat=img)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

def cropFace():
    dim = (500, 500)
    counter = 0
    xbuffer = 40
    ybuffer = 40
    for filename in os.listdir("Aligned"):
        if filename.endswith(".jpg"):
            imgName = os.path.join("Aligned", filename)
            img = cv2.imread(imgName)
            gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            top = -999999
            bottom = 999999
            for face in faces:
                x1 = face.left() # left point
                y1 = face.top() # top point
                x2 = face.right() # right point
                y2 = face.bottom()

                landmarks = predictor(image=gray, box=face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                if (n == 0):
                    left = x;
                elif (n == 16):
                    right = x;
                elif (n == 24 or n == 19 or n == 25 or n == 20 or n == 23 or n == 18 and y > top):
                    top = y;
                elif (n == 8 or n == 7 or n == 9 and y < bottom):
                    bottom = y;
            if((left - xbuffer) < 0):
                left = 0
            else:
                left = left - xbuffer

            if ((top - ybuffer) < 0):
                top = 0
            else:
                top = top - ybuffer

            right = right + xbuffer
            bottom = bottom + ybuffer
            faceImage = img[top:bottom, left:right]

            filename = os.path.join("Cropped", str(counter))
            filename = filename + ".jpg"

            finalFace = cv2.resize(faceImage, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(filename, finalFace)
            counter = counter + 1

def getEyeGroups(data):
    data2 = np.array(data)
    kmeans = KMeans(n_clusters=4).fit(data2)
    cluster = kmeans.fit_predict(data2)
    dictionary = {}
    for index in range(len(data)):
        if cluster[index] in dictionary:
            value = []
            value = dictionary[cluster[index]]
            value.append(data[index])
            dictionary[cluster[index]] = value
        else:
            dictionary[cluster[index]] = data[index]
    print("Group 0: ", dictionary[0])
    print("Group 1: ", dictionary[1])
    print("Group 2: ", dictionary[2])
    print("Group 3: ", dictionary[3])
    dictionary = {}
    for index in range(len(data)):
        if cluster[index] in dictionary:
            value = []
            value = dictionary[cluster[index]]
            value.append(picFile[index])
            dictionary[cluster[index]] = value
        else:
            dictionary[cluster[index]] = [picFile[index]]
    dir = "EyeGroup0/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = "EyeGroup1/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = "EyeGroup2/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = "EyeGroup3/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    for file in dictionary[0]:
        img = cv2.imread(file)
        filepath = os.path.join("EyeGroup0", file)
        cv2.imwrite(filepath, img)
    for file in dictionary[1]:
        img = cv2.imread(file)
        filepath = os.path.join("EyeGroup1", file)
        cv2.imwrite(filepath, img)
    for file in dictionary[2]:
        img = cv2.imread(file)
        filepath = os.path.join("EyeGroup2", file)
        cv2.imwrite(filepath, img)
    for file in dictionary[3]:
        img = cv2.imread(file)
        filepath = os.path.join("EyeGroup3", file)
        cv2.imwrite(filepath, img)

def getMouthGroups(data):
    data2 = np.array(data)
    kmeans = KMeans(n_clusters=4).fit(data2)
    cluster = kmeans.fit_predict(data2)
    dictionary = {}
    for index in range(len(data)):
        if cluster[index] in dictionary:
            value = []
            value = dictionary[cluster[index]]
            value.append(data[index])
            dictionary[cluster[index]] = value
        else:
            dictionary[cluster[index]] = data[index]
    print("Group 0: ", dictionary[0])
    print("Group 1: ", dictionary[1])
    print("Group 2: ", dictionary[2])
    print("Group 3: ", dictionary[3])
    dictionary = {}
    for index in range(len(data)):
        if cluster[index] in dictionary:
            value = []
            value = dictionary[cluster[index]]
            value.append(picFile[index])
            dictionary[cluster[index]] = value
        else:
            dictionary[cluster[index]] = [picFile[index]]
    dir = "MouthGroup0/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = "MouthGroup1/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = "MouthGroup2/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir = "MouthGroup3/Cropped/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    for file in dictionary[0]:
        img = cv2.imread(file)
        filepath = os.path.join("MouthGroup0", file)
        cv2.imwrite(filepath, img)
    for file in dictionary[1]:
        img = cv2.imread(file)
        filepath = os.path.join("MouthGroup1", file)
        cv2.imwrite(filepath, img)
    for file in dictionary[2]:
        img = cv2.imread(file)
        filepath = os.path.join("MouthGroup2", file)
        cv2.imwrite(filepath, img)
    for file in dictionary[3]:
        img = cv2.imread(file)
        filepath = os.path.join("MouthGroup3", file)
        cv2.imwrite(filepath, img)

#cropFace()
#eyeDistAll()
mouthDistAll()