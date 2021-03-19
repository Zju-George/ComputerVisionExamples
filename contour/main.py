import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils import *

import warnings

warnings.filterwarnings('ignore')


class QuadDetector(object):
    def __init__(self, img, show=True):
        self.img = img
        self.show = show
        self.height, self.width = self.img.shape[:2]

        self.VLINE1, self.VLINE2, self.HLINE1, self.HLINE2 = self.initFixedLines()
        self.vLines, self.hLines = self.initQuadLines()
        self.getFourLines()

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
        black = gray.copy()
        black[:, :] = np.zeros(1, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:4]
        cv2.drawContours(black, contours, -1, 255, 3)
        cannyEdges = cv2.Canny(black, 50, 150)
        # group vertical and horizontal lines
        lines = cv2.HoughLinesP(cannyEdges, 1, np.pi / 180, 100, maxLineGap=200)
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

        if self.show:
            cv2.imshow('contours', black)
            cv2.imshow('canny edges', cannyEdges)
        return vLines, hLines

    def getFourLines(self):
        top1, bottom1 = self.intersectVLINE(self.VLINE1)
        top2, bottom2 = self.intersectVLINE(self.VLINE2)

        if self.show:
            self.drawPoint(top1, color=(0, 0, 255), thickness=-1)
            self.drawPoint(top2, color=(0, 0, 255), thickness=-1)
            self.drawPoint(bottom1, color=(0, 0, 255), thickness=-1)
            self.drawPoint(bottom2, color=(0, 0, 255), thickness=-1)

    def intersectVLINE(self, VLINE):
        # TODO: use hLines to intersect with VLINE
        crossPoints = []
        for line in self.hLines:
            point = GetCrossPoint(line, VLINE)
            if self.show:
                self.drawPoint(point, color=(0, 255, 0))
            crossPoints.append([point.x, point.y])
        crossPoints = np.array(crossPoints)
        kMeans = KMeans(n_clusters=2, random_state=0).fit(crossPoints)
        topPoint = Point(kMeans.cluster_centers_[0][0], kMeans.cluster_centers_[0][1])
        bottomPoint = Point(kMeans.cluster_centers_[1][0], kMeans.cluster_centers_[1][1])
        return topPoint, bottomPoint

    def intersectHLINE(self, HLINE):
        # TODO: use vLines to intersect with HLINE
        pass

    def drawLine(self, line):
        cv2.line(self.img, (int(line.p1.x), int(line.p1.y)), (int(line.p2.x), int(line.p2.y)), (0, 255, 0), 1)
        return

    def drawPoint(self, point, color=(0, 0, 255), thickness=1):
        cv2.circle(self.img, (int(point.x), int(point.y)), 3, color, thickness=thickness)
        return

    def hang(self):
        if self.show:
            for vLine in self.vLines:
                self.drawLine(vLine)
            for hLine in self.hLines:
                self.drawLine(hLine)
            cv2.imshow('img', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    img = cv2.imread("./butterfly.png")
    img = cv2.imread("./bf.png")

    # -------------------------------
    quadDetector = QuadDetector(img)

    quadDetector.hang()
    return
    # -------------------------------
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # get the [1, 4] largest area contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:2]
    # cv2.drawContours(img, contours, -1, (0,0,255), 3)

    # get the edges from contours
    black = gray.copy()
    black[:, :] = np.zeros(1, dtype=np.uint8)
    cv2.drawContours(black, contours, -1, 255, 3)
    edges = cv2.Canny(black, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, maxLineGap=200)
    print(f'lines number: {len(lines)}')

    VLINE = Line(Point(0, 0), Point(0, height))
    HLINE = Line(Point(0, 0), Point(width, 0))
    vLines = []
    hLines = []

    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        grad = (y2 - y1) / (x2 - x1)
        line = Line(Point(x1, y1), Point(x2, y2))
        line.grad = grad
        if np.abs(grad) < 1:
            hLines.append(line)
        else:
            vLines.append(line)

    cps = []
    for line in hLines:
        cp = GetCrossPoint(line, VLINE)
        cps.append([cp.x, cp.y])
        print(f"Cross point: {(cp.x, cp.y)}")
    cps = np.array(cps)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(cps)
    print(f'cross points: {cps} label: {kmeans.labels_}')
    argsort = np.argsort(cps[:, 1])
    topLine, bottomLine = hLines[argsort[1]], hLines[argsort[-3]]
    drawLine(img, topLine)
    drawLine(img, bottomLine)
    cv2.imshow("img", img)
    # cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
