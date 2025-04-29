import cv2
import numpy as np

# define color ranges
color_hsv_ranges = {
            'red': {'low': [150, 120, 100], 'high': [180, 255, 255]},
            'green': {'low': [63, 120, 90], 'high': [85, 200, 255]},
            'blue': {'low': [99, 93, 73], 'high': [105, 255, 255]},
            'yellow': {'low': [22, 180, 179], 'high': [27, 254, 255]},
            'orange': {'low': [5, 120, 157], 'high': [13, 209, 255]},
            'purple': {'low': [115, 29, 40], 'high': [147, 112, 100]}
        }

# read image
image = cv2.imread('/Users/guofucius/Documents/Umich/ROB 550/output.jpg')  # 确保路径正确
if image is None:
    print("Error: Image not found or failed to load. Please check the file path.")
    exit()

# convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

kernel = np.ones((5, 5), np.uint8)

for color, hsv_range in color_hsv_ranges.items():
    # define color range
    lower_bound = np.array(hsv_range['low'])
    upper_bound = np.array(hsv_range['high'])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150:  
            x, y, w, h = cv2.boundingRect(contour)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            size = "Large" if area > 780 else "Small"

            # calculate aspect ratio and angle of the contour
            rect = cv2.minAreaRect(contour)
            width = rect[1][0]
            height = rect[1][1]
            ratio = width / height

            flag = 0.5 < ratio < 2
            angle = 0 if abs(rect[2]) < 0.5 or abs(rect[2] - 90) < 0.5 else rect[2]
    
            if flag:
                #cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)  # draw center point
            
                # add label to the image
                label = f"{color.capitalize()} ({size})"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f"Center: ({center_x}, {center_y})", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f"Angle: {angle:.2f}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# show the detected objects in the image
cv2.imshow('Detected Objects', image)

# push q to quit the program
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
