import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--draw', action="store_true")
parser.add_argument('--nomask', action="store_true")
parser.add_argument('--ref', type=str, default='../assets/butterfly.png')
parser.add_argument('--scan', type=str, default='../assets/scan1.jpg')
args = parser.parse_args()

MAX_MATCHES = 1000
GOOD_MATCH_PERCENT = 0.6

def pltImshow(title, img):
  plt.title(title)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.show()
  return

def alignImages(imgScan, imgRef):
  print(f'Alignment configs:\nMAX_MATCHES={MAX_MATCHES}\nGOOD_MATCH_PERCENT={GOOD_MATCH_PERCENT}')

  # Convert images to grayscale
  imgScanGray = cv2.cvtColor(imgScan, cv2.COLOR_BGR2GRAY)
  imgRefGray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors
  orb = cv2.ORB_create(MAX_MATCHES)
  kpsScan, desScan = orb.detectAndCompute(imgScanGray, None)
  kpsRef, desRef = orb.detectAndCompute(imgRefGray, None)
  
  # Match features
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
  matches = matcher.match(desScan, desRef, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Using only TOP-numGoodMatches matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imgMatches = cv2.drawMatches(imgScan, kpsScan, imgRef, kpsRef, matches, None)

  # Extract location of good matches
  pointsScan = np.zeros((len(matches), 2), dtype=np.float32)
  pointsRef = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    pointsScan[i, :] = kpsScan[match.queryIdx].pt
    pointsRef[i, :] = kpsRef[match.trainIdx].pt
  
  # Find homography. Possible flags: cv2.RANSAC
  homographyMatrix, mask = cv2.findHomography(pointsScan, pointsRef, cv2.RHO) 
  matchesMask = mask.ravel().tolist()

  # Draw only inliers matches
  if not args.nomask:
    imgMatches = cv2.drawMatches(imgScan, kpsScan, imgRef, kpsRef, matches, None, matchesMask=matchesMask)

  # Use homography to transform the image, also the target image size
  height, width, channels = imgRef.shape
  imgAlign = cv2.warpPerspective(imgScan, homographyMatrix, (width, height))
  
  return imgAlign, homographyMatrix, imgMatches


if __name__ == '__main__':
  # Read reference image
  print(f'Reading reference image: {args.ref}')
  imgRef = cv2.imread(args.ref, cv2.IMREAD_COLOR)

  # Read scanned image to be aligned
  print(f'Reading image to align: {args.scan}')
  imgScan = cv2.imread(args.scan, cv2.IMREAD_COLOR)
  
  imgAlign, homographyMatrix, imgMatches = alignImages(imgScan, imgRef)
  # print("Estimated homography: \n",  homographyMatrix)

  # Save aligned image to disk.
  outFilename = "aligned.jpg"
  print("Saving aligned image: ", outFilename)
  cv2.imwrite(outFilename, imgAlign)

  # Stack imgAlign & imgRef
  if args.draw:
    pltImshow('Matches', imgMatches)
    pltImshow('Aligned & Reference', np.hstack((imgAlign, imgRef)))
  
