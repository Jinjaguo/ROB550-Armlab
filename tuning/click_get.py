import cv2
import numpy as np

# golbal variables
drawing = False  # drawing flag
start_point = (-1, -1)  
end_point = (-1, -1)  

# callback function for mouse events
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # start drawing
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:  # update drawing
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:  # finish drawing
        drawing = False
        end_point = (x, y)

        # calculate the selected region's HSV values
        x1, y1 = start_point
        x2, y2 = end_point
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        
        # extract the selected region's HSV values
        hsv_crop = hsv[y1:y2, x1:x2]

        min_hsv = hsv_crop.min(axis=(0, 1))
        max_hsv = hsv_crop.max(axis=(0, 1))

        print(f"Selected Region HSV Min: {min_hsv}, Max: {max_hsv}")

image = cv2.imread('/Users/guofucius/Documents/Umich/ROB 550/output.jpg')  
if image is None:
    print("Error: Image not found or failed to load. Please check the file path.")
    exit()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_rectangle)

while True:
    img_copy = image.copy()

    if drawing:
        cv2.rectangle(img_copy, start_point, end_point, (255, 0, 0), 2)

    cv2.imshow('Image', img_copy)

    # push 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all windows and exit
cv2.destroyAllWindows()
