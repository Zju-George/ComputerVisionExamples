import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils import *

import argparse
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--draw', action="store_true")
parser.add_argument('--img', type=str, default='assets/scan1.jpg')
parser.add_argument('--savePath', type=str, default='corners.txt')


class QuadDetector(object):
    def __init__(self, img, draw=True):
        self.img = img
        self.draw = draw
        self.height, self.width = self.img.shape[:2]

        self.VLINE1, self.VLINE2, self.HLINE1, self.HLINE2 = self.initFixedLines()
        self.vLines, self.hLines = self.initQuadLines()
        self.topLeft, self.bottomLeft, self.bottomRight, self.topRight = self.getQuadCorners()

    def getCoordinates(self):
        # inverse clock
        coordinates = [str(self.topLeft.x), str(self.topLeft.y),
                       str(self.bottomLeft.x), str(self.bottomLeft.y),
                       str(self.bottomRight.x), str(self.bottomRight.y),
                       str(self.topRight.x), str(self.topRight.y)]
        return coordinates

    def initFixedLines(self):
        VLINE1 = Line(Point(0, 0), Point(0, self.height))
        VLINE2 = Line(Point(self.width, 0), Point(self.width, self.height))
        HLINE1 = Line(Point(0, 0), Point(self.width, 0))
        HLINE2 = Line(Point(0, self.height), Point(self.width, self.height))
        return VLINE1, VLINE2, HLINE1, HLINE2

    def initQuadLines(self):
        # process image
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contourImg = gray.copy()
        contourImg[:, :] = np.zeros(1, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:4]
        cv2.drawContours(contourImg, contours, -1, 255, 3)
        cannyImg = cv2.Canny(contourImg, 50, 150)
        # group vertical and horizontal lines
        lines = cv2.HoughLinesP(cannyImg, 1, np.pi / 180, 100, minLineLength=300, maxLineGap=100)
        print(f'HoughLinesP number: {len(lines)}')
        vLines = []
        hLines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            grad = (y2 - y1) / (x2 - x1)
            line = Line(Point(x1, y1), Point(x2, y2))
            line.grad = grad
            if np.abs(grad) < 1:
                hLines.append(line)
            else:
                vLines.append(line)
        if self.draw:
            cv2.imshow('contours', contourImg)
            cv2.imshow('canny edges', cannyImg)
        return vLines, hLines

    def getQuadCorners(self):
        top1, bottom1 = self.intersectVLINE(self.VLINE1)
        top2, bottom2 = self.intersectVLINE(self.VLINE2)
        left1, right1 = self.intersectHLINE(self.HLINE1)
        left2, right2 = self.intersectHLINE(self.HLINE2)

        topLine = Line(top1, top2)
        bottomLine = Line(bottom1, bottom2)
        leftLine = Line(left1, left2)
        rightLine = Line(right1, right2)
        topLeft = GetCrossPoint(topLine, leftLine)
        topRight = GetCrossPoint(topLine, rightLine)
        bottomLeft = GetCrossPoint(bottomLine, leftLine)
        bottomRight = GetCrossPoint(bottomLine, rightLine)

        if self.draw:
            self.drawPoint(top1, color=(0, 0, 255), thickness=-1)
            self.drawPoint(top2, color=(0, 0, 255), thickness=-1)
            self.drawPoint(bottom1, color=(0, 0, 255), thickness=-1)
            self.drawPoint(bottom2, color=(0, 0, 255), thickness=-1)
            self.drawLine(topLine, color=(0, 0, 255))
            self.drawLine(bottomLine, color=(0, 0, 255))
            self.drawLine(leftLine, color=(0, 0, 255))
            self.drawLine(rightLine, color=(0, 0, 255))
            self.drawPoint(topLeft, color=(0, 0, 255), thickness=-1, size=5)
            self.drawPoint(topRight, color=(0, 0, 255), thickness=-1, size=5)
            self.drawPoint(bottomLeft, color=(0, 0, 255), thickness=-1, size=5)
            self.drawPoint(bottomRight, color=(0, 0, 255), thickness=-1, size=5)

        return topLeft, bottomLeft, bottomRight, topRight

    def intersectVLINE(self, VLINE):
        # use hLines to intersect with VLINE
        crossPoints = []
        for line in self.hLines:
            point = GetCrossPoint(line, VLINE)
            if self.draw:
                self.drawPoint(point, color=(0, 255, 0))
            crossPoints.append([point.x, point.y])
        crossPoints = np.array(crossPoints)
        kMeans = KMeans(n_clusters=2, random_state=0).fit(crossPoints)
        topPoint = Point(kMeans.cluster_centers_[0][0], kMeans.cluster_centers_[0][1])
        bottomPoint = Point(kMeans.cluster_centers_[1][0], kMeans.cluster_centers_[1][1])
        if topPoint.y > bottomPoint.y:
            temp = topPoint
            topPoint = bottomPoint
            bottomPoint = temp
        return topPoint, bottomPoint

    def intersectHLINE(self, HLINE):
        # use vLines to intersect with HLINE
        crossPoints = []
        for line in self.vLines:
            point = GetCrossPoint(line, HLINE)
            if self.draw:
                self.drawPoint(point, color=(0, 255, 0))
            crossPoints.append([point.x, point.y])
        crossPoints = np.array(crossPoints)
        kMeans = KMeans(n_clusters=2, random_state=0).fit(crossPoints)
        leftPoint = Point(kMeans.cluster_centers_[0][0], kMeans.cluster_centers_[0][1])
        rightPoint = Point(kMeans.cluster_centers_[1][0], kMeans.cluster_centers_[1][1])
        if leftPoint.x > rightPoint.x:
            temp = leftPoint
            leftPoint = rightPoint
            rightPoint = temp
        return leftPoint, rightPoint

    def drawLine(self, line, color=(0, 255, 0)):
        cv2.line(self.img, (int(line.p1.x), int(line.p1.y)), (int(line.p2.x), int(line.p2.y)), color, 1)
        return

    def drawPoint(self, point, color=(0, 0, 255), thickness=1, size=3):
        cv2.circle(self.img, (int(point.x), int(point.y)), size, color, thickness=thickness)
        return

    def hang(self):
        if self.draw:
            for vLine in self.vLines:
                self.drawLine(vLine)
            for hLine in self.hLines:
                self.drawLine(hLine)
            cv2.imshow('img', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parser.parse_args()
    img = cv2.imread(args.img)

    quadDetector = QuadDetector(img, draw=args.draw)
    coordinates = quadDetector.getCoordinates()
    quadDetector.hang()

    result = '|'.join(coordinates)
    print(f'Corners coordinates: {result}')
    with open(args.savePath, 'w') as file:
        file.write(result)


if __name__ == '__main__':
    main()
