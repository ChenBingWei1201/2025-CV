import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    # Step 1: Create a Difference_of_Gaussian object with the threshold value from args
    # - Function: Difference_of_Gaussian(args.threshold)
    DoG = Difference_of_Gaussian(args.threshold)

    # Step 2: Get keypoints from the input image
    # - Function: get_keypoints(img)
    keypoints = DoG.get_keypoints(img)

    # Step 3: Plot keypoints on the input image and save it
    # - Function: plot_keypoints(img_gray, keypoints, save_path)
    plot_keypoints(img, keypoints, 'output/1_out.png')

if __name__ == '__main__':
    main()