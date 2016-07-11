import cv2
import numpy as np 

def nothing(x):
	pass


cap = cv2.VideoCapture(0)

cv2.namedWindow('video')


#Creating the trackbar
cv2.createTrackbar('H', 'video', 0, 179, nothing)

#Some default values (For yellow smiley ball)
FINAL_MIN = np.array([30-10, 50, 50])
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





cap.release()
cv2.destroyAllWindows()