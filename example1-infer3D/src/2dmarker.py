import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='pnp.png')
parser.add_argument('--save', action="store_true")

def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = '%d,%d' % (x, y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (0, 0, 255), thickness = 1)
        cv2.imshow('image', img)


if __name__ == '__main__':
    args = parser.parse_args()

    img = cv2.imread(f'../assets/{args.img}')
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse)
    k = cv2.waitKey(0)
    if k == 27:
        if args.save==True:
            cv2.imwrite('2dmarker.png', img)
            print('2dmarker.png saved')
        else:
            print('not save')
        cv2.destroyAllWindows()