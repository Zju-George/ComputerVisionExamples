import cv2
import numpy as np
import argparse
import time

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--draw', action="store_true")

class ReconstructionData(object):
    def __init__(self, camera_matrix=None, distortion_coeffs=None, model_points=None, image_points=None):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.model_points = model_points
        self.image_points = image_points
        assert len(self.model_points) == len(self.image_points)
        if len(self.model_points)<=4:
            Logger.error('model points less than 4, cannot solve properly!')
        else:
            Logger.info('reconstruction data initialized.')


class Reconstruction(object):
    def __init__(self, data=None, learning_rate=0.000002, thrsh=5, max_steps=100, image=None, draw=False):
        self.data = data
        self.learning_rate = learning_rate
        self.thrsh = thrsh
        self.max_steps = max_steps
        self.image = image
        self.draw = draw

        self.rotation_vector, self.translation_vector = self.solve_pnp()
        self.target2D = np.zeros(2)
        self.coordinate3D = np.zeros(3)
        self.loss = 0.
        

    def l2_loss(self, point1, point2):
        return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2
    
    def compute_grad(self):
        epsilon = 0.00001
        delta_x_3D = self.coordinate3D + epsilon * np.array([1., 0., 0.])
        point2D, _ = cv2.projectPoints(delta_x_3D,
        self.rotation_vector, self.translation_vector, self.data.camera_matrix, self.data.distortion_coeffs)
        loss = self.l2_loss(point2D.reshape(-1), self.target2D)
        grad_x = (loss-self.loss)/epsilon

        delta_y_3D = self.coordinate3D + epsilon * np.array([0., 1., 0.])
        point2D, _ = cv2.projectPoints(delta_y_3D,
        self.rotation_vector, self.translation_vector, self.data.camera_matrix, self.data.distortion_coeffs)
        loss = self.l2_loss(point2D.reshape(-1), self.target2D)
        grad_y = (loss-self.loss)/epsilon

        return np.array([grad_x, grad_y])

    def solve_pnp(self):
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.data.model_points, 
        self.data.image_points, self.data.camera_matrix, self.data.distortion_coeffs, flags=8)
        # Logger.debug(f'rotation_vector:\n {rotation_vector}')
        # Logger.debug(f'translation_vector:\n {translation_vector}')
        return rotation_vector, translation_vector

    def init_guess(self):
        # TODO: bilinear interpolation to init coordinate3D, now just init as (0., 0., 0.)
        self.coordinate3D = np.array([0., 0., 0.])

        point2D, _ = cv2.projectPoints(self.coordinate3D, 
        self.rotation_vector, self.translation_vector, self.data.camera_matrix, self.data.distortion_coeffs)
        self.loss = self.l2_loss(point2D.reshape(-1), self.target2D)
    
    def opt_step(self):
        grad = self.compute_grad()
        grad = np.append(grad, 0.)
        self.coordinate3D = self.coordinate3D + (-grad) * self.learning_rate
        point2D, _ = cv2.projectPoints(self.coordinate3D,
        self.rotation_vector, self.translation_vector, self.data.camera_matrix, self.data.distortion_coeffs)
        point2D = point2D.reshape(-1)
        self.loss = self.l2_loss(point2D, self.target2D)
        # Logger.debug(f'loss: {self.loss}')
        if self.draw:
            cv2.circle(self.image, (int(point2D[0]), int(point2D[1])), 2, (0, 255, 0), thickness = 2)
            cv2.imshow('image', self.image)
        return
        
    def opt(self, target2D):
        start = time.time()
        self.target2D = target2D
        self.init_guess()
        steps = 0
        for i in range(self.max_steps):
            self.opt_step()
            steps += 1
            if self.loss < self.thrsh:
                break
        end = time.time()
        Logger.info(f'coordinate3D result: {self.coordinate3D}; loss: {self.loss}; optimization steps: {steps}; time cost: {end-start}s')

    def hangon(self):
        if not self.draw:
            return
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parser.parse_args()

    # original data to prepare 
    camera_matrix = np.array([[618.41368969, 0., 325.36183392], [0., 622.17832864, 264.46629453], [0., 0., 1.]], dtype='double')
    distortion_coeffs =  np.array([[ 3.72960294e-02, -1.56467602e-02, -3.25651528e-04, 1.03897830e-03]], dtype='double')
    model_points = np.array([(0., 0., 0.), (1.199, 0., 0.), (0.197, 0.088, 0.467), 
                            (0.304, 0.088, 0.467), (0.304, 0.088, 0.337), (0.197, 0.088, 0.337)])
    image_points = np.array([(92, 395), (488, 417), (73, 444), (116, 447), (139, 423), (98, 419)], dtype='double')

    # init data class and reconstruction class
    data = ReconstructionData(camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs, model_points=model_points, image_points=image_points)
    image = cv2.imread('../assets/pnp.png')
    reconstruction = Reconstruction(data=data, image=image, draw=args.draw)

    target2D = np.array([252, 257], dtype='double')
    reconstruction.opt(target2D)
    # optimize function
    reconstruction.hangon()
    