from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings
import os
import cv2
import pytesseract
def signin(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(username=email, password=password)

        if user is not None:
            login(request=request, user= user)
            return redirect('home')
        else:
            return HttpResponse("Invalid mail or password")
        
    return render(request, 'signin.html')

def signup(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        username = request.POST.get('name')
        pass1 = request.POST.get('pass1')
        pass2 = request.POST.get('pass2')
        first_name,last_name = username.split(' ')
        if pass1 != pass2: 
            return HttpResponse("Incorrect Password")
        else:   
            user = User.objects.create_user(email,email=email, password = pass1)
            user.first_name = first_name
            user.last_name = last_name

            user.save()
            return redirect('signin')

    return render(request, 'signup.html')

@login_required(login_url= "/")
def home(request):
    return render(request, 'dashboard.html')

@login_required(login_url= "/")
def service(request):
    return render(request, 'services.html')

@login_required(login_url= "/")
def about(request):
    return render(request, 'about.html')

@login_required(login_url= "/")
def contact(request):
    return render(request, 'contact.html')

@login_required(login_url= "/")
def profile(request):
    return render(request, 'profile.html')

@login_required(login_url= "/")
def logout_user(request):
    logout(request)
    return redirect('/')

@login_required(login_url= "/")
def userlist(request):
    users = User.objects.exclude(username=request.user.username)

    return render(request, 'userlist.html', {'users': users})


@login_required(login_url= "/")
def update(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')

        User.objects.filter(id = request.user.id ).update(first_name = first_name, last_name = last_name, email = email)

        return redirect('profile')

    return render(request, 'update.html')

def imagebased(request):
    return render(request,"imagebased.html")


@login_required(login_url= "/")
def imagebased1(request):
    import base64
    uploaded_file = request.FILES['image']
    import cv2
    import pytesseract
    import numpy as np
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
    # Read the image file

    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("Original", image)

    # Convert to Grayscale Image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Canny Edge Detection
    canny_edge = cv2.Canny(gray_image, 170, 200)
    # Find contours based on Edges
    contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
    # Initialize license Plate contour and x,y,w,h coordinates
    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None
    # Find the contour with 4 potential corners and create ROI around it
    for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4:                                            #see whether it is a Rect
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h, x:x + w]
            break
    (thresh, license_plate) = cv2.threshold(license_plate, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow("plate",license_plate)
    # Removing Noise from the detected image, before sending to Tesseract
    license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
    (thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

    #Text Recognition
    text = pytesseract.image_to_string(license_plate)
    #Draw License Plate and write the Text
    
    # Draw License Plate and write the Text
    image_with_detection = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 0, 255), 3)
    image_with_detection = cv2.putText(image_with_detection, text, (x - 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert images to base64 format
    _, original_image_encoded = cv2.imencode('.png', image)
    original_image_base64 = base64.b64encode(original_image_encoded).decode('utf-8')
    
    _, detected_image_encoded = cv2.imencode('.png', image_with_detection)
    detected_image_base64 = base64.b64encode(detected_image_encoded).decode('utf-8')

    # Pass original and detected images to the template
    context = {
        'original_image': original_image_base64,
        'detected_image': detected_image_base64,
        'license_plate': text,
    }

    return render(request, 'imagebased.html', context)


def videobased(request):
    return render(request,"videobased.html")


from django.shortcuts import render
from django.http import HttpResponse
import os
import cv2
import pytesseract

def videobased1(request):
    if request.method == 'POST' and request.FILES.get('video'):
        upload = request.FILES['video']
        temp_file_path = os.path.join(settings.MEDIA_ROOT, upload.name)
        
        # Save the uploaded file to a temporary location on the server
        with open(temp_file_path, 'wb+') as destination:
            for chunk in upload.chunks():
                destination.write(chunk)

        # Load the Haar cascade classifier for license plate detection
        plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

        # Load the video file
        video = cv2.VideoCapture(temp_file_path)

        if not video.isOpened():
            print('Error Reading Video')
            return HttpResponse("Error: Could not open video.")

        detected_plates = []

        while True:
            ret, frame = video.read()    
            if not ret:
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect license plates in the frame
            plates = plat_detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))

            for (x, y, w, h) in plates:
                # Draw a rectangle around the detected license plate
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Extract the region of interest (ROI) containing the license plate
                roi = gray_frame[y:y+h, x:x+w]

                # Perform OCR on the license plate region to recognize the text
                license_plate_text = pytesseract.image_to_string(roi, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                
                # Draw the recognized text on the frame
                cv2.putText(frame, license_plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Print the recognized license plate text
                print("Detected License Plate:", license_plate_text)
                detected_plates.append(license_plate_text)

            # Display the frame with license plate detection and recognition
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object
        video.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Delete the temporary file
        os.remove(temp_file_path)

        return render(request, 'videobased.html', {'detected_plates': detected_plates})
