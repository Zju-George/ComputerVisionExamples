import cv2  
 
img = cv2.imread("./butterfly.png")
img = cv2.imread("./bf.png")  
 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  

contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# get first 5 large contour
contours=sorted(contours, key = cv2.contourArea, reverse = True)[1:5]

for contour in contours:
    # 画直方图
    print(f'contour shape: {contour.shape}')
    # print(contour)
    print('---------')
cv2.drawContours(img, contours, -1, (0,0,255), 3)  

cv2.imshow("img", img)  
cv2.waitKey(0)  