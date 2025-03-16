import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        # Apply gaussian blur with corresponding sigma value on the input.
        # The base image of the 1st octave is the input image.
        # Down sample the last blurred image in the 1st octave as the base image
        # of the 2nd octave.
        gaussian_images = []

        for i in range(self.num_octaves):
            if i == 0:
                base_image = image
            else:
                height, width = gaussian_images[0][-1].shape[:2]
                base_image = cv2.resize(gaussian_images[0][-1], (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)

            octave_images = [base_image]
            for j in range(1, self.num_guassian_images_per_octave):
                blurred_image = cv2.GaussianBlur(base_image, (0, 0), self.sigma ** j)
                octave_images.append(blurred_image)

            gaussian_images.append(octave_images)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        # You should subtract the second image (more blurred one) to the first
        # image (less blurred one) to get DoG.
        dog_images = []

        for i in range(self.num_octaves):
            octave_images = gaussian_images[i]
            dog_sub = []
            for j in range(self.num_DoG_images_per_octave):
                sub = cv2.subtract(octave_images[j+1], octave_images[j])
                dog_sub.append(sub)
                cv2.imwrite(f"output/DoG{i+1}-{j+1}.png", sub)
            
            dog_images.append(dog_sub)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        # Find the local extremum and threshold the pixel value.
        keypoints = []

        for i in range(self.num_octaves):
            octave_dog = dog_images[i]
            for j in range(1, self.num_DoG_images_per_octave - 1):  # Ignore first & last DoG
                for y in range(1, octave_dog[j].shape[0] - 1):
                    for x in range(1, octave_dog[j].shape[1] - 1):
                        value = octave_dog[j][y, x]
                        if np.abs(value) < self.threshold:
                            continue  # Ignore weak keypoints
                        
                        # Extract 3x3x3 neighborhood
                        neighbors = np.concatenate([
                            np.array(octave_dog[j - 1][y-1:y+2, x-1:x+2]).flatten(),  # Below
                            np.array(octave_dog[j][y-1:y+2, x-1:x+2]).flatten(),      # Same
                            np.array(octave_dog[j + 1][y-1:y+2, x-1:x+2]).flatten()   # Above
                        ])
                        
                        if value >= np.max(neighbors) or value <= np.min(neighbors):
                            if i == 0:
                                keypoints.append((y, x))
                            else:
                                keypoints.append((y*2, x*2))

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis=0)
        
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
