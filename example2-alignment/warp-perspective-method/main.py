import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15

def pltImshow(title, img):
  plt.title(title)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.show()
  return

def alignImages(imgScan, imgRef):

  # Convert images to grayscale
  imgScanGray = cv2.cvtColor(imgScan, cv2.COLOR_BGR2GRAY)
  imgRefGray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_MATCHES)
  keypoints1, descriptors1 = orb.detectAndCompute(imgScanGray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(imgRefGray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches & show
  imgMatches = cv2.drawMatches(imgScan, keypoints1, imgRef, keypoints2, matches, None)
  pltImshow('Matches', imgMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  homographyMatrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography to transform the image, also the target image size
  height, width, channels = imgRef.shape
  im1Reg = cv2.warpPerspective(imgScan, homographyMatrix, (width, height))
  
  return im1Reg, homographyMatrix


if __name__ == '__main__':
  dir = '../assets'

  # Read reference image
  # imgRef = os.path.join(dir, 'form.jpg')
  imgRef = os.path.join(dir, 'butterfly.png')
  print('Reading reference image: ', imgRef)
  imgRef = cv2.imread(imgRef, cv2.IMREAD_COLOR)

  # Read scanned image to be aligned
  # imgScan = os.path.join(dir, 'scanned-form.jpg')
  imgScan = os.path.join(dir, 'scan1.jpg')
  print('Reading image to align: ', imgScan);  
  imgScan = cv2.imread(imgScan, cv2.IMREAD_COLOR)
  
  print('Aligning images ...')
  imgAlign, h = alignImages(imgScan, imgRef)
  # print("Estimated homography: \n",  h)

  # Save aligned image to disk.
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imgAlign)

  # stack imgRef & imgAlign
  pltImshow('Reference & Aligned', np.hstack((imgRef, imgAlign)))
  
