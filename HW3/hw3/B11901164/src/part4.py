import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = dst.copy()

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        
        # Brute-force matcher with Hamming distance for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        # TODO: 2. apply RANSAC to choose best H
        best_H = None
        max_inliers = 0
        inlier_threshold = 5.0  # Pixel distance threshold
        
        # Number of iterations
        ransac_iterations = 200
        min_samples = 4  # Minimum points needed for homography
        
        for _ in range(ransac_iterations):
            # Randomly select 4 matching points
            if len(matches) < min_samples:
                continue
                
            indices = random.sample(range(len(matches)), min_samples)
            sample_src = src_pts[indices]
            sample_dst = dst_pts[indices]
            
            # Calculate homography using these 4 points - H maps from img2 to img1
            H = solve_homography(sample_dst, sample_src)
            
            # Count inliers
            inliers = 0
            for i in range(len(matches)):
                # Convert destination point to homogeneous coordinates
                p2 = np.array([dst_pts[i][0], dst_pts[i][1], 1.0])
                
                # Apply homography to get the projected point in img1
                p1_proj = H @ p2
                
                # Convert back to inhomogeneous coordinates
                p1_proj = p1_proj / p1_proj[2]
                
                # Compute the distance
                dist = np.sqrt((p1_proj[0] - src_pts[i][0])**2 + (p1_proj[1] - src_pts[i][1])**2)
                
                if dist < inlier_threshold:
                    inliers += 1
            
            # Update the best model if this one has more inliers
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H
        
        if best_H is None:
            # If RANSAC failed, use minimal set of matches
            best_H = solve_homography(dst_pts[:min_samples], src_pts[:min_samples])

        # TODO: 3. chain the homographies
        # best_H maps from img2 to img1
        # last_best_H maps from img1 to panorama
        # Thus, last_best_H @ best_H maps from img2 to panorama
        current_H = last_best_H @ best_H
        
        # Update for next iteration
        last_best_H = current_H

        # TODO: 4. apply warping
        # Warp the new image to the panorama
        out = warping(im2, out, current_H, 0, h_max, 0, w_max, direction='b')

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)