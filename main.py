import cv2
import numpy as np 

SPIKE_FACTOR = 5

#to see if hit
touch = 0
center_touch = (1,1)

def nothing(x):
	pass

class Derivative(object):

	def __init__(self):

		self.x = []
		self.y = []
		self.der = []

	def findDer(self):
		
		if len(self.x) > 1:

			if self.x[-1] != self.x[-2]:

				self.der.append((self.y[-1] - self.y[-2])/(self.x[-1] - self.x[-2]))

			else:
				self.der.append(10000)

		if len(self.der) > 4:
			self.der.pop(1)

		self.checkHit()

	def checkHit(self):

		if len(self.der) == 4:

			avg = 0

			for i in range(1,4):

				avg += self.der[-i - 1]

			avg /= 3

			if self.der[-1]/avg >= 5:

				print 1
				touch = 1
				center_touch = (x,y)

	def add(self, x, y):

		self.x.append(x)
		self.y.append(y)

		if len(self.x) >2:

			self.x.pop(1)
			self.y.pop(1)

		self.findDer()



hit = Derivative()

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
x_arr = []
y_arr = []


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
		
		print 1
		FINAL_MIN = COLOR_MIN
		FINAL_MAX = COLOR_MAX	
		break

#To detect the ball

prev_radius = 0

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
		x_arr.append(x)
		y_arr.append(y)
		
		M = cv2.moments(c)
		
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
		cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
		
		cv2.circle(frame, center, 5, (0, 0, 255), -1)
		
		#Derivative thing
		hit.add(x,y)

		'''
		#Old Approach
		if radius - prev_radius > radius*0.2:
			touch = 1
			center_touch = center
		'''


		prev_radius = radius


	if touch:
		cv2.circle(frame, center_touch, 10, (255, 0, 0), -1)

	cv2.imshow('video', frame)
	cv2.imshow('detail',frame_threshold)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break




cap.release()
cv2.destroyAllWindows()


x_arr = np.array(map(lambda x:float(x),x_arr))
y_arr = np.array(map(lambda x:float(x),y_arr))

y_der = (y_arr[1:]-y_arr[:-1])/(x_arr[1:]-x_arr[:-1])

import matplotlib.pyplot as plt
plt.plot(x_arr,y_arr)
plt.plot(y_der)
plt.show()
