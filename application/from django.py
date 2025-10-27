from django.http import HttpResponse
from django.shortcuts import render
import pymysql
from django.shortcuts import render
from recommend.models import Attendance
from recommend.models import user
import pymysql
import datetime
import numpy as np
import cv2
import os
from PIL import Image, ImageTk
from django.db import transaction
from django.contrib.auth import get_user_model, authenticate, login

User = get_user_model()

count = 0
cap = cv2.VideoCapture(0)
button_path = None
newpath, id_input, label1, button = None, None, None, None

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('C:/Project2023-24/Smart Attendance System (Image Processing)/sas/Recommendationsystem/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.2, 5)

    if faces == ():
        return False
    else:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

def take_pic():
    global count

    face_cascade = cv2.CascadeClassifier('C:/Project2023-24/Smart Attendance System (Image Processing)/sas/Recommendationsystem/haarcascade_frontalface_default.xml')

    while (True):
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from the camera.")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if faces == ():
            return False
        else:
            for (x, y, w, h) in faces:
                crop_img = gray[y:y+h, x:x+w]
                count += 1
                cv2.imwrite(newpath + "/User_" + str(id_input) + '.' + str(count) + ".jpg", crop_img)

        if count == 50:
            label1.config(text="Please smile and capture.")
            count += 1
            return False
        elif count == 100:
            label1.config(text="Please take off glasses if they exist and capture.")
            count += 1
            return False
        elif count == 150:
            label1.config(text="Please try to draw a circle with your head")
            count += 1
            return False
        elif count > 200:
            label1.config(text="Done! You can close the program.")
            train_faces(newpath, id_input)
            return False

def train_faces(newpath, id_input):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('C:/Project2023-24/Smart Attendance System (Image Processing)/sas/Recommendationsystem/haarcascade_frontalface_default.xml')

    get_imagePath = [os.path.join(newpath, f) for f in os.listdir(newpath)]
    faceSamples = []
    ids = []

    for imagePath in get_imagePath:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    recognizer.train(faceSamples, np.array(ids))
    recognizer.save('trained/' + id_input + '.yml')

def show_frame():
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from the camera.")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detect_face(rgb)

    prevImg = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=prevImg)
def train(request):
    return render(request,"train.html")

def train1(request):
    count = 0
    cap = cv2.VideoCapture(0)

    # Initialize newpath outside of the if block
    newpath = None

    if request.method == 'POST':
        id_input = request.POST.get('id_input')
        print(id_input)
        newpath = r'face_data/' + id_input
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        button = "Capture"
        label1 = "Please capture when you are ready!"

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier('C:/Project2023-24/Smart Attendance System (Image Processing)/sas/Recommendationsystem/haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            if faces == ():
                continue
                
            for (x, y, w, h) in faces:
                gray = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

            crop_img = gray[y:y+h, x:x+w]
            count += 1
            cv2.imwrite(newpath + "/User_" + str(id_input) + '.' + str(count) + ".jpg", crop_img)

            if count == 20:
                label1 = "Please smile and capture."
                print(label1)
            elif count == 50:
                label1 = "Please take off glasses if they exist and capture."
                print(label1)
            elif count > 100:
                label1 = "Done! Student Enrollement done"
                print(label1)
                train_faces(newpath, id_input)
                break

        return render(request, 'train.html', {'label1':label1})

    return render(request, 'train.html', {'label1':label1})

mydb=pymysql.connect(host="localhost",user="root",password="root",database="crop")

def page1(request):
    return render(request,"index.html")
def userhome(request):
    return render(request,"userdashboard.html")
def aboutus(request):
    return render(request,"aboutus.html")
def login(request):
    return render(request,"login.html")

def logout(request):
    return render(request,"login.html")

def register(request):
    return render(request,"register.html")

def adminhome(request):
    return render(request,"admindashboard.html")


def viewuser(request):
    content={}
    payload=[]
    q1="select * from user";
    cur=mydb.cursor()
    cur.execute(q1)
    res=cur.fetchall()
    for x in res:
        content={'name':x[0],"contact":x[1],"email":x[2],"uid":x[4]}
        payload.append(content)
        content={}
    return render(request,"viewuser.html",{'list': {'items':payload}})


def doremove(request):
    uid= request.GET.get("uid")
    q1=" delete from user where uid=%s";
    values=(uid,)
    cur=mydb.cursor()
    cur.execute(q1,values)
    mydb.commit()
    return viewuser(request)

def prevpred(request):
    content={}
    payload=[]
    uid=request.session['uid']
    q1="select * from smp where uid=%s";
    values=(uid)
    cur=mydb.cursor()
    cur.execute(q1,values)
    res=cur.fetchall()
    for x in res:
        content={'N':x[0],"P":x[1],"K":x[2],"T":x[3],'H':x[4],"ph":x[5],"rainfall":x[6],"pred":x[8]}
        payload.append(content)
        content={}
    return render(request,"prevpred.html",{'list': {'items':payload}})

def viewpredicadmin(request):
    content={}
    payload=[]
    q1="select * from user";
    cur=mydb.cursor()
    cur.execute(q1)
    res=cur.fetchall()
    for x in res:
        content={'name':x[0],"contact":x[1],"email":x[2],"uid":x[4]}
        payload.append(content)
        content={}
    return render(request,"prevpredadmin.html",{'list': {'items':payload}})
    
def dologin(request):
    sql="select * from user";
    cur=mydb.cursor()
    cur.execute(sql)
    data=cur.fetchall()
    email=request.POST.get('email')
    password=request.POST.get('password')
    name="";    
    uid="";
    isfound="0";
    content={}
    payload=[]
    print(email)
    print(password)
    if(email=="admin" and password=="admin"):
        print("print")
        return render(request,"admindashboard.html")
    else:
        for x in data:
           if(x[2]==email and x[3]==password):
               request.session['uid']=x[4]
               request.session['name']=x[0]
               request.session['contact']=x[1]
               request.session['email']=x[2]
               request.session['pass']=x[3]
               isfound="1"
        if(isfound=="1"):
            return render(request,"userdashboard.html")
        else:
            return render(request,"error.html")


def doregister(request):
    name=request.POST.get('name')
    contact=request.POST.get('contact')
    email=request.POST.get('email')
    password=request.POST.get('password')
    sql="INSERT INTO user(name,contact,email,password) VALUES (%s,%s,%s,%s)";
    values=(name,contact,email,password)
    cur=mydb.cursor()
    cur.execute(sql,values)
    mydb.commit()
    return render(request,"login.html")
    
    

def attendance(request):
    return render(request, "attendance.html")



    
def takeattendance(request):
    # It helps in identifying the faces
    import cv2, sys, numpy, os
    size = 4
    #haar_file = 'haarcascade_frontalface_default.xml'
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    datasets = 'C:/Project2023-24/Smart Attendance System (Image Processing)/sas/face_data'
    ##path = 'trained/'

    # Part 1: Create fisherRecognizer
    print('Recognizing Face Please Be in sufficient Lights...')

    # Create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    images = [cv2.resize(image, (width, height)) for image in images]
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)

    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(0)
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            eyes = eye_cascade.detectMultiScale(gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(im,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
      
            if prediction[1]<70:
               student_name = names[prediction[0]]
               time = datetime.datetime.now()

               
               sql = "INSERT INTO attendance (name, time) VALUES (%s, %s)"
    
               values=(student_name,time)
               cur=mydb.cursor()
               cur.execute(sql,values)
               mydb.commit()
               print("Current time:", time)    
               print("New Attendance object created:", student_name)
    
               cv2.putText(im, '% s - %.0f' % (student_name, prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
              cv2.putText(im, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
      
        cv2.imshow('OpenCV', im)
          
        key = cv2.waitKey(10)
        if key == 27:
            break

    return render(request,"attendance.html")