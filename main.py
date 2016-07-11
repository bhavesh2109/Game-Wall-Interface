import cv2
import numpy as np 

def nothing(x):
	pass


cap = cv2.VideoCapture(0)

cv2.namedWindow('video')
cv2.namedWindow('detail')

#Creating the trackbar
cv2.createTrackbar('H', 'video', 10, 179, nothing)

cv2.createTrackbar('param2', 'detail', 0, 200, nothing)

#Some default values (For yellow smiley ball)
FINAL_MIN = np.array([30-10, 100, 100])
FINAL_MAX = np.array([30+10, 255, 255])


#To accept the value for H that is suitable for the ball


while(1):

	ret, frame = cap.read()

	#Input from trackbar
	h = cv2.getTrackbarPos('H', 'video')
	
	#Min, Max colors according to hue chosen
	COLOR_MIN = np.array([h-10, 100, 100])
	COLOR_MAX = np.array([h+10, 255, 255])

	#Change to HSV
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


	#Threshold according to value
	frame_threshold = cv2.inRange(hsv_frame, COLOR_MIN, COLOR_MAX)

	#Display the image
	cv2.imshow('video', frame_threshold)

	#To go to the default value
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	#If the color is chosen by the user
	if cv2.waitKey(1) & 0xFF == ord('y'):
		
		FINAL_MIN = COLOR_MIN
		FINAL_MAX = COLOR_MAX	
		break

#To detect the ball
while(1):
	
	ret, frame = cap.read()

	#Change to HSV
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#Threshold according to values 
	frame_threshold = cv2.inRange(hsv_frame, FINAL_MIN, FINAL_MAX)

	frame_threshold = cv2.medianBlur(frame_threshold, 5)

	frame_threshold = cv2.erode(frame_threshold, None, iterations = 2)
	frame_threshold = cv2.dilate(frame_threshold, None, iterations = 2)

	#Hough circle transform
	
	cnts = cv2.findContours(frame_threshold.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	center = None

	if len(cnts) > 0:
		
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		
		M = cv2.moments(c)
		
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
		cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
		
		cv2.circle(frame, center, 5, (0, 0, 255), -1)
	

	cv2.imshow('video', frame)
	cv2.imshow('detail',frame_threshold)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break




cap.release()
cv2.destroyAllWindows()