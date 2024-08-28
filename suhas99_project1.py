# -*- coding: utf-8 -*-
"""
# ENPM673 Project-1 Part-1

> By: Suhas Nagaraj

> Directory ID: suhas99


Problem 1:

In the provided video footage, a black object is thrown against a wall. Your objective is to develop a Python script to detect
and plot the pixel coordinates of the center point of this thrown object throughout the video. Follow the steps outlined
below:

1) Read the video and extract individual frames using OpenCV. [15]

2) Loop over each frame to extract the pixels of moving object (Hint: Use color). [20]

3) Calculate the centroid of the object in every frame (doesn’t have to be very precise). [15]

4) Assume TOP LEFT corner of the frame as 0,0 and accordingly use ‘Standard Least Square’ to fit a curve (parabola) through the found centroids in part 3. [20]

5) Given that x axis value is 1000, find the y axis value for calculated equation in part 4. [10]

6) Capture any one frame from the video (which shows the object) and plot the obtained equation. [20]

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# creating a video file object
vid=cv.VideoCapture("project1_video1.mp4")

# Condition statement to check if the video is opened and loaded, without errors
if vid.isOpened():
  print("Video is Opened",'\n')
else:
  print("Error")

"""
The code above is for creating a video object and checking if the video is opened for further operations
"""

# Initializing the lists used to store the centroid values
x, y = [], []

# Creating an index to extract a video frame for plotting
f=0

# Loop which runs as long as the video is opened
while vid.isOpened():

  # retrieving each frame from the video
  ret, frame = vid.read()

  # condition statment to check if the retrieval is successful.
  # When the frames are retrieved, ret is true.
  # After all the frames of the video are retreived, the ret becomes false and this condition is used to break out of the loop
  if ret==False:
    break

  # Condition statement to extract frame 300 from the video for plotting
  if f==300:
    cv.imwrite("plotting_frame.jpg", frame)
    print("Done extracting frame 300")

  # Converting the image/frame to Greyscale for easier object detection
  gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

  # Extracting all pixles whose value is less than 20
  pixels = np.where(gray_image < 20)

  f=f+1

  # Condition to check if there are pixels with value less than 20
  if len(pixels[0]) > 0:

    # Calculating the centroid by taking mean with respect to row and column values
    rc = np.mean(pixels[0])
    cc = np.mean(pixels[1])

    # Storing the centroid values in list for plotting
    y.append(rc)
    x.append(cc)

"""
The code above is for reading each frame of video and checking if the desired object is in the frame. If the object is present, its centroid is calculated and is stored in lists.
"""

# Calculating the required parameters required for graph fitting
sx=sum(x)
sy=sum(y)
n=len(x)
sx2=sum([p * q for p, q in zip(x, x)])
sx3=sum([p * q * r  for p, q , r in zip(x, x, x)])
sx4=sum([p * q * r * s  for p, q , r , s in zip(x, x, x, x)])
sxy=sum([p * q for p, q in zip(x, y)])
sx2y=sum([p * q * r  for p, q , r in zip(x, x, y)])


# Creating numpy arrays representing the linear equations mentioned below
C = np.array([[sx2, sx, n],[sx3, sx2, sx],[sx4, sx3, sx2]])
r = np.array([sy, sxy, sx2y])

# Solving the system of linear equations to get the value of constants a, b and c
a,b,c = np.linalg.solve(C, r)

print("In the equation ( y=(ax^2)+(bx)+c ), the a,b and c values are: a = ",a,", b = ",b,", c = ",c)

"""The above cell represents the fitting the center points into a parabola (  **y=(a*x^2)+(b*x)+c** ) using "Standard Least Square" technique using following equations

1) sum(y) = a * sum(x^2) + b * sum(x) +  n * c

2) sum(x * y) = a * sum(x^3) + b * sum(x^2) + c * sum(x)

3) sum((x^2) * y) = a * sum(x^4) + b * sum(x^3) + c * sum(x^2)

Note: These equations are derived by taking the least square of the distances between the points and the parabola (represented by E)
"""

# Plotting the curve using matplotlib
x_parabola = np.linspace(min(x), max(x))
y_parabola = (a * x_parabola * x_parabola) + (b * x_parabola) + c
plt.scatter(x, y, label='Center Points')
plt.plot(x_parabola, y_parabola, color='red', label='Parabola Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.gca().invert_yaxis()
plt.title('Parabola Fit for Center Points')
plt.legend()
plt.grid(True)
plt.show()

"""The parabola plot above represents a least square fit of the extracted centroid points of the object. The parabola represents the trajectory of the object."""

x_given = 1000

# Calculating y for x=1000
y_calc = (a * x_given * x_given) + (b * x_given) + c
print("For x=1000, the corresponding y value is: ", y_calc)

"""
Calculating the y value for x = 1000
"""

# Reading the image saved earlier (frame 300)
img = plt.imread("plotting_frame.jpg")
fig, ax = plt.subplots()
ax.imshow(img)

#Plotting the parabola fitted curve on the image
ax.plot(x_parabola, y_parabola, color='red')
plt.savefig('path.jpg', bbox_inches='tight')
"""The trajectory of the object is plotted over the image extracted earlier. The red represents the trajectory"""
