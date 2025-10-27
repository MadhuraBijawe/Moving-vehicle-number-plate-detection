from django.shortcuts import render,redirect, HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from django.conf import settings


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

import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from skimage import measure
import imutils
import os

def sort_cont(character_contours):
	"""
	To sort contours
	"""
	i = 0
	boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
	
	(character_contours, boundingBoxes) = zip(*sorted(zip(character_contours,
														boundingBoxes),
													key = lambda b: b[1][i],
													reverse = False))
	
	return character_contours


def segment_chars(plate_img, fixed_width):
	
	"""
	extract Value channel from the HSV format
	of image and apply adaptive thresholding
	to reveal the characters on the license plate
	"""
	V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

	thresh = cv2.adaptiveThreshold(V, 255,
								cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								cv2.THRESH_BINARY,
								11, 2)

	thresh = cv2.bitwise_not(thresh)

	# resize the license plate region to
	# a canoncial size
	plate_img = imutils.resize(plate_img, width = fixed_width)
	thresh = imutils.resize(thresh, width = fixed_width)
	bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

	# perform a connected components analysis
	# and initialize the mask to store the locations
	# of the character candidates
	labels = measure.label(thresh, background = 0)

	charCandidates = np.zeros(thresh.shape, dtype ='uint8')

	# loop over the unique components
	characters = []
	for label in np.unique(labels):
		
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask to display
		# only connected components for the current label,
		# then find contours in the label mask
		labelMask = np.zeros(thresh.shape, dtype ='uint8')
		labelMask[labels == label] = 255

		cnts = cv2.findContours(labelMask,
					cv2.RETR_EXTERNAL,
					cv2.CHAIN_APPROX_SIMPLE)

		cnts = cnts[1] if imutils.is_cv3() else cnts[0]

		# ensure at least one contour was found in the mask
		if len(cnts) > 0:

			# grab the largest contour which corresponds
			# to the component in the mask, then grab the
			# bounding box for the contour
			c = max(cnts, key = cv2.contourArea)
			(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

			# compute the aspect ratio, solodity, and
			# height ration for the component
			aspectRatio = boxW / float(boxH)
			solidity = cv2.contourArea(c) / float(boxW * boxH)
			heightRatio = boxH / float(plate_img.shape[0])

			# determine if the aspect ratio, solidity,
			# and height of the contour pass the rules
			# tests
			keepAspectRatio = aspectRatio < 1.0
			keepSolidity = solidity > 0.15
			keepHeight = heightRatio > 0.5 and heightRatio < 0.95

			# check to see if the component passes
			# all the tests
			if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
				
				# compute the convex hull of the contour
				# and draw it on the character candidates
				# mask
				hull = cv2.convexHull(c)

				cv2.drawContours(charCandidates, [hull], -1, 255, -1)

	contours, hier = cv2.findContours(charCandidates,
										cv2.RETR_EXTERNAL,
										cv2.CHAIN_APPROX_SIMPLE)
	
	if contours:
		contours = sort_cont(contours)
		
		# value to be added to each dimension
		# of the character
		addPixel = 4
		for c in contours:
			(x, y, w, h) = cv2.boundingRect(c)
			if y > addPixel:
				y = y - addPixel
			else:
				y = 0
			if x > addPixel:
				x = x - addPixel
			else:
				x = 0
			temp = bgr_thresh[y:y + h + (addPixel * 2),
							x:x + w + (addPixel * 2)]

			characters.append(temp)
			
		return characters
	
	else:
		return None



class PlateFinder:
	def __init__(self, minPlateArea, maxPlateArea):
		
		# minimum area of the plate
		self.min_area = minPlateArea
		
		# maximum area of the plate
		self.max_area = maxPlateArea 

		self.element_structure = cv2.getStructuringElement(
							shape = cv2.MORPH_RECT, ksize =(22, 3))

	def preprocess(self, input_img):
		
		imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
		
		# convert to gray..
		gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
		
		# sobelX to get the vertical edges
		sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3) 
		
		# otsu's thresholding
		ret2, threshold_img = cv2.threshold(sobelx, 0, 255,
						cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		element = self.element_structure
		morph_n_thresholded_img = threshold_img.copy()
		cv2.morphologyEx(src = threshold_img,
						op = cv2.MORPH_CLOSE,
						kernel = element,
						dst = morph_n_thresholded_img)
		
		return morph_n_thresholded_img

	def extract_contours(self, after_preprocess):
		
		contours, _ = cv2.findContours(after_preprocess,
										mode = cv2.RETR_EXTERNAL,
										method = cv2.CHAIN_APPROX_NONE)
		return contours

	def clean_plate(self, plate):
		
		gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray,
									255,
									cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
									cv2.THRESH_BINARY,
									11, 2)
		
		contours, _ = cv2.findContours(thresh.copy(),
										cv2.RETR_EXTERNAL,
										cv2.CHAIN_APPROX_NONE)

		if contours:
			areas = [cv2.contourArea(c) for c in contours]
			
			# index of the largest contour in the area
			# array
			max_index = np.argmax(areas) 

			max_cnt = contours[max_index]
			max_cntArea = areas[max_index]
			x, y, w, h = cv2.boundingRect(max_cnt)
			rect = cv2.minAreaRect(max_cnt)
			if not self.ratioCheck(max_cntArea, plate.shape[1],
												plate.shape[0]):
				return plate, False, None
			
			return plate, True, [x, y, w, h]
		
		else:
			return plate, False, None



	def check_plate(self, input_img, contour):
		
		min_rect = cv2.minAreaRect(contour)
		
		if self.validateRatio(min_rect):
			x, y, w, h = cv2.boundingRect(contour)
			after_validation_img = input_img[y:y + h, x:x + w]
			after_clean_plate_img, plateFound, coordinates = self.clean_plate(
														after_validation_img)
			
			if plateFound:
				characters_on_plate = self.find_characters_on_plate(
											after_clean_plate_img)
				
				if (characters_on_plate is not None and len(characters_on_plate) == 8):
					x1, y1, w1, h1 = coordinates
					coordinates = x1 + x, y1 + y
					after_check_plate_img = after_clean_plate_img
					
					return after_check_plate_img, characters_on_plate, coordinates
		
		return None, None, None



	def find_possible_plates(self, input_img):
		
		"""
		Finding all possible contours that can be plates
		"""
		plates = []
		self.char_on_plate = []
		self.corresponding_area = []

		self.after_preprocess = self.preprocess(input_img)
		possible_plate_contours = self.extract_contours(self.after_preprocess)

		for cnts in possible_plate_contours:
			plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
			
			if plate is not None:
				plates.append(plate)
				self.char_on_plate.append(characters_on_plate)
				self.corresponding_area.append(coordinates)

		if (len(plates) > 0):
			return plates
		
		else:
			return None

	def find_characters_on_plate(self, plate):

		charactersFound = segment_chars(plate, 400)
		if charactersFound:
			return charactersFound

	# PLATE FEATURES
	def ratioCheck(self, area, width, height):
		
		min = self.min_area
		max = self.max_area

		ratioMin = 3
		ratioMax = 6

		ratio = float(width) / float(height)
		
		if ratio < 1:
			ratio = 1 / ratio
		
		if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
			return False
		
		return True

	def preRatioCheck(self, area, width, height):
		
		min = self.min_area
		max = self.max_area

		ratioMin = 2.5
		ratioMax = 7

		ratio = float(width) / float(height)
		
		if ratio < 1:
			ratio = 1 / ratio

		if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
			return False
		
		return True

	def validateRatio(self, rect):
		(x, y), (width, height), rect_angle = rect

		if (width > height):
			angle = -rect_angle
		else:
			angle = 90 + rect_angle

		if angle > 15:
			return False
		
		if (height == 0 or width == 0):
			return False

		area = width * height
		
		if not self.preRatioCheck(area, width, height):
			return False
		else:
			return True
class OCR:
	
	def __init__(self, modelFile, labelFile):
		
		self.model_file = modelFile
		self.label_file = labelFile
		self.label = self.load_label(self.label_file)
		self.graph = self.load_graph(self.model_file)
		self.sess = tf.compat.v1.Session(graph=self.graph, 
										config=tf.compat.v1.ConfigProto())

	def load_graph(self, modelFile):
		
		graph = tf.Graph()
		graph_def = tf.compat.v1.GraphDef()
		
		with open(modelFile, "rb") as f:
			graph_def.ParseFromString(f.read())
		
		with graph.as_default():
			tf.import_graph_def(graph_def)
		
		return graph

	def load_label(self, labelFile):
		label = []
		proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
		
		for l in proto_as_ascii_lines:
			label.append(l.rstrip())
		
		return label

	def convert_tensor(self, image, imageSizeOuput):
		"""
		takes an image and transform it in tensor
		"""
		image = cv2.resize(image,
						dsize =(imageSizeOuput,
								imageSizeOuput),
						interpolation = cv2.INTER_CUBIC)
		
		np_image_data = np.asarray(image)
		np_image_data = cv2.normalize(np_image_data.astype('float'),
									None, -0.5, .5,
									cv2.NORM_MINMAX)
		
		np_final = np.expand_dims(np_image_data, axis = 0)
		
		return np_final

	def label_image(self, tensor):

		input_name = "import/input"
		output_name = "import/final_result"

		input_operation = self.graph.get_operation_by_name(input_name)
		output_operation = self.graph.get_operation_by_name(output_name)

		results = self.sess.run(output_operation.outputs[0],
								{input_operation.outputs[0]: tensor})
		results = np.squeeze(results)
		labels = self.label
		top = results.argsort()[-1:][::-1]
		
		return labels[top[0]]
	

	def label_image_list(self, listImages, imageSizeOuput):
		plate = ""
		
		for img in listImages:
			
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
			plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput))
		
		return plate, len(plate)
	

import os
import cv2
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
# Assuming PlateFinder and OCR are defined somewhere in your application
# from application.plate_recognition import PlateFinder, OCR

def draw_plate_number(img, plate_number):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)  # Green color
    lineType = 2

    cv2.putText(img, plate_number, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

def videobased1(request):
    if request.method == 'POST' and request.FILES.get('video'):
        upload = request.FILES['video']
        temp_file_path = os.path.join(settings.MEDIA_ROOT, upload.name)
        
        # Save the uploaded file to a temporary location on the server
        with open(temp_file_path, 'wb+') as destination:
            for chunk in upload.chunks():
                destination.write(chunk)
        
        # Initialize PlateFinder and OCR objects
        findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)
        model = OCR(modelFile=r"E:\Vaishnavi\moving no plate detection\CoreProject\application\model\binary_128_0.50_ver3.pb",
                    labelFile=r"E:\Vaishnavi\moving no plate detection\CoreProject\application\model\binary_128_0.50_labels_ver2.txt")
        
        # Open the video file from the temporary location
        cap = cv2.VideoCapture(temp_file_path)
        
        # Set window name and resize window
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 400, 300)  # Set small frame size
        
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                # Resize the frame
                resized_img = cv2.resize(img, (400, 300))
                
                # Process the frame for plate recognition
                possible_plates = findPlate.find_possible_plates(img)
                if possible_plates is not None:
                    for i, p in enumerate(possible_plates):
                        chars_on_plate = findPlate.char_on_plate[i]
                        recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                        print("Recognized plate:", recognized_plate)
                        draw_plate_number(resized_img, recognized_plate)

                cv2.imshow("Video", resized_img)  # Display the frame
                if cv2.waitKey(1) & 0xFF == 27:  # Press 'esc' key to exit
                    break
            else:
                break
        
        # Release video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()
        
        # Delete the temporary file
        os.remove(temp_file_path)
        
        # No need to print recognized plates here if you're showing them in the frame
        # Return a response (you might want to redirect to another page)
        return HttpResponse("Video processing complete.")

    return render(request, 'your_template.html')
