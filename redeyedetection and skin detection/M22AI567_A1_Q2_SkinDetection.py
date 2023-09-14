#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import numpy as np

def main():
    
    input_image = cv2.imread("handfolding.jpg")

    # Converting from BGR image to HSV color space images
    hsv_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    # image Skin color range for HSV color space image
    hsv_imagemask = cv2.inRange(hsv_img, (0, 15, 0), (17, 170, 255))
    hsv_imagemask = cv2.morphologyEx(hsv_imagemask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Converting image from BGR to YCbCr color space image
    ycrcb_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
    
    # Skin color range for YCbCr color space
    ycrcb_imagemask = cv2.inRange(ycrcb_img, (0, 135, 85), (255, 180, 135))
    ycrcb_imagemask = cv2.morphologyEx(ycrcb_imagemask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Merge skin detection results (YCbCr and HSV)
    global_mask = cv2.bitwise_and(ycrcb_imagemask, hsv_imagemask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    # Invert the masks for visualization
    hsv_Imageresult = cv2.bitwise_not(hsv_imagemask)
    ycrcb_Imageresult = cv2.bitwise_not(ycrcb_imagemask)
    global_Imageresult = cv2.bitwise_not(global_mask)

    # Save the result images
    cv2.imwrite("1_HSV.jpg", hsv_Imageresult)
    cv2.imwrite("2_YCbCr.jpg", ycrcb_Imageresult)
    cv2.imwrite("3_global_result.jpg", global_Imageresult)

    # Read and resize the result images to display horizontally
    img1 = cv2.imread('1_HSV.jpg')
    img1 = cv2.resize(img1, dsize=(0, 0), fx=0.3, fy=0.3)
    img2 = cv2.imread('2_YCbCr.jpg')
    img2 = cv2.resize(img2, dsize=(0, 0), fx=0.3, fy=0.3)
    img3 = cv2.imread('3_global_result.jpg')
    img3 = cv2.resize(img3, dsize=(0, 0), fx=0.3, fy=0.3)
    input_image = cv2.resize(input_image, dsize=(0, 0), fx=0.3, fy=0.3)

    # Concatenate images horizontally for visualization
    concatenated_image = cv2.hconcat([input_image, img1, img2, img3])
    
    # Display the concatenated image
    cv2.imshow('Skin Detection Results', concatenated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




