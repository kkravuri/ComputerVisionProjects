#!/usr/bin/env python
# coding: utf-8

# In[66]:


# importing Packages
import cv2

# loading pre-trained classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def detect_faces(img, gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    total_faces = len(faces)
    faces_present = total_faces > 0
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return faces, faces_present, total_faces

def detect_eyes(img, gray, face):
    (x, y, w, h) = face
    roi_gray_face = gray[y:y + h, x:x + w]
    roi_color_face = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(
        roi_gray_face, 1.1, 5, minSize=(20, 20), maxSize=(80, 80))
    total_eyes = len(eyes)
    for eye in eyes:
        (ex, ey, ew, eh) = eye
        # Calculate eye coordinates relative to the whole image
        
        eye_x = ex + x
        eye_y = ey + y
        # Extract the eye region
        eye_region = img[eye_y:eye_y+eh, eye_x:eye_x+ew]
        # Convert the eye region to HSV color space
        hsv_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        # Define lower and upper thresholds for detecting red color
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        # Create a mask for red regions in the eye
        red_mask = cv2.inRange(hsv_eye, lower_red, upper_red)
        
        # Calculate the percentage of red pixels
        total_pixels = red_mask.shape[0] * red_mask.shape[1]
        red_pixel_count = np.sum(red_mask == 255)
        red_percentage = red_pixel_count / total_pixels
        
        red_threshold=0.01
            
        if red_percentage >= red_threshold:
            print("Red-eye detected in the eye region.")
        else:
            print("No red-eye detected in the eye region.")
                   
        cv2.rectangle(roi_color_face, (ex, ey),(ex + ew, ey + eh), (0, 255, 0), 2)
        
    return eyes, total_eyes

#read image
img = cv2.imread("gagarin_2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces, faces_present, total_faces = detect_faces(img, gray)
print(total_faces)
if faces_present:
    for face in faces:
        eyes, total_eyes = detect_eyes(img, gray, face)
    print('eyes count',total_eyes)
    
    # Display image
    cv2.imshow('Red Eye marked Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('No Face detected')
    cv2.imshow('No face Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



# In[ ]:




