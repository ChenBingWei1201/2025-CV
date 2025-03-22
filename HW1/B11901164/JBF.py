import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = int(3*sigma_s)
        
        # Precompute spatial kernel
        x = np.arange(-self.pad_w, self.pad_w+1)
        y = np.arange(-self.pad_w, self.pad_w+1)
        xx, yy = np.meshgrid(x, y)
        self.spatial_kernel = np.exp(-(xx**2 + yy**2) / (2 * self.sigma_s**2))
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # Handle multi-channel vs single-channel guidance
        if len(guidance.shape) == 2:  # Single channel
            is_single_channel = True
        else:  # Multi-channel
            is_single_channel = False
        
        # Output setup
        output = np.zeros_like(img, dtype=np.float64)  # Use float64 for higher precision
        h, w = img.shape[:2]
        
        # Normalize guidance to [0, 1]
        padded_guidance = padded_guidance / 255.0
        
        # Iterate over each pixel in the original image
        for i in range(h):
            for j in range(w):
                # Extract the local window
                i_pad, j_pad = i + self.pad_w, j + self.pad_w
                win_i = slice(i_pad - self.pad_w, i_pad + self.pad_w + 1)
                win_j = slice(j_pad - self.pad_w, j_pad + self.pad_w + 1)
                
                img_window = padded_img[win_i, win_j]
                
                # Calculate the range kernel based on guidance
                if is_single_channel:  # Single channel
                    guidance_window = padded_guidance[win_i, win_j]
                    center_val = padded_guidance[i_pad, j_pad]
                    
                    # Calculate intensity difference and apply Gaussian
                    diff = (guidance_window - center_val)**2
                    range_kernel = np.exp(-diff / (2 * self.sigma_r**2))
                else:  # Multi-channel (RGB)
                    guidance_window = padded_guidance[win_i, win_j]
                    center_val = padded_guidance[i_pad, j_pad]
                    
                    # Calculate squared Euclidean distance between vectors
                    diff = np.sum((guidance_window - center_val)**2, axis=2)
                    range_kernel = np.exp(-diff / (2 * self.sigma_r**2))
                
                # Combine spatial and range kernels
                weight = self.spatial_kernel * range_kernel
                
                # Reshape weight for broadcasting with multi-channel images
                if len(img.shape) == 3:
                    weight = weight[:, :, np.newaxis]
                
                # Apply weight to image window and normalize
                weighted_sum = np.sum(img_window * weight, axis=(0, 1))
                weight_sum = np.sum(weight)
                
                output[i, j] = weighted_sum / weight_sum
        
        return np.clip(output, 0, 255).astype(np.uint8)
