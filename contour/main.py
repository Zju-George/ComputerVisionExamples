import cv2  
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def analyzeContour(contour):
    contour = contour.reshape((-1, 2))
    # print(f'contour shape: {contour.shape}\n-----')
    for i in range(contour.shape[0]-1):
        # print(f'current: {contour[i]} next: {contour[i+1]}')
        pass
    pass

img = cv2.imread("./butterfly.png")
img = cv2.imread("./bf.png")  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# get the [1, 4] largest contours
contours=sorted(contours, key = cv2.contourArea, reverse = True)[1:5]

# get the edges from contours
black = gray.copy()
black[:, :]=np.zeros(1, dtype=np.uint8)
cv2.drawContours(black, contours, -1, 255, 3)
edges = cv2.Canny(black, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, maxLineGap=100)
print(lines.shape)
for line in lines:
    x1, y1, x2, y2 = line[0]
    
    grad = (y2-y1)/(x2-x1)
    print(f'grad: {grad}')
    if grad < -60: 
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # cv2.circle(img, (x1, y1), 5, (0, 0, 255))
        # cv2.circle(img, (x2, y2), 5, (0, 0, 255))

for contour in contours:
    #TODO: Hough line detection
    # https://www.youtube.com/watch?v=KEYzUP7-kkU
    analyzeContour(contour)
    
# cv2.drawContours(img, contours, -1, (0,0,255), 3)

cv2.imshow("img", img)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()