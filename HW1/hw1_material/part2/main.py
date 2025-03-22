import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # read setting file
    sigma_s = 0
    sigma_r = 0.0
    rgb_coefficients = []

    with open(args.setting_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3 and parts[0] != 'R':
                rgb_coefficients.append(list(map(float, parts)))
            elif len(parts) == 4:
                sigma_s = float(parts[1])
                sigma_r = float(parts[3])
    # plot grayscale images
    img_grays = [img_gray]
    for i in range(len(rgb_coefficients)):
        coeff = rgb_coefficients[i]
        img_gray_transformed = (img_rgb[:, :, 0] * coeff[0] + 
                                img_rgb[:, :, 1] * coeff[1] + 
                                img_rgb[:, :, 2] * coeff[2]).astype(np.uint8)
        img_grays.append(img_gray_transformed)

    for i in range(len(img_grays)):
        cv2.imwrite(f'./output/1/1_gray_{i}.png', img_grays[i])

    # create an instance of Joint_bilateral_filter
    jbf = Joint_bilateral_filter(sigma_s, sigma_r)

    # apply bilateral filter
    bf_out = jbf.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    costs = []

    # calculate the cost
    for i in range(len(img_grays)):
        jbf_out = jbf.joint_bilateral_filter(img_rgb, img_grays[i]).astype(np.uint8)
        cv2.imwrite(f'./output/1/1_rgb_{i}.png',  cv2.cvtColor(jbf_out,cv2.COLOR_RGB2BGR))
        cost = np.sum(np.abs(jbf_out.astype(np.int32)-bf_out.astype(np.int32)))
        costs.append(cost)

    print(f"Costs: {costs}")
    print(f"Highest cost: {max(costs)} and index: {costs.index(max(costs))}")
    print(f"Lowest cost: {min(costs)} and index: {costs.index(min(costs))}")

if __name__ == '__main__':
    main()