# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []
    
    # Define the target size for tiny images (8x8 works better than 16x16 for scene recognition)
    target_size = (8, 8)
    
    for img_path in img_paths:
        # Load the image
        img = Image.open(img_path)
        
        # First crop the center square portion to maintain aspect ratio
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        # Crop center square and convert to grayscale
        img = img.crop((left, top, right, bottom)).convert('L')
        
        # Resize to target_size
        resized_img = img.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array and flatten
        img_array = np.array(resized_img).flatten()
        
        # Normalize the image (zero mean and unit length)
        img_mean = np.mean(img_array)
        img_array = img_array - img_mean
        
        # L2 normalization (unit length)
        img_norm = np.linalg.norm(img_array)
        if img_norm > 0:
            img_array = img_array / img_norm
        
        tiny_img_feats.append(img_array)
    
    # Convert to numpy array
    tiny_img_feats = np.array(tiny_img_feats)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    
    # List to collect all SIFT features
    all_features = []
    
    # Sample a subset of training images (every 5th image)
    sampled_img_paths = img_paths[::5]
    
    # Process each image
    for img_path in tqdm(sampled_img_paths, desc="Extracting SIFT features"):
        # Load and convert the image to grayscale
        img = Image.open(img_path).convert('L')
        img_array = np.array(img).astype('float32')
        
        # Extract SIFT features with larger step size for speed
        _, descriptors = dsift(img_array, step=[5, 5], fast=True)
        
        # If no features were found, skip this image
        if descriptors.shape[0] == 0:
            continue
            
        # Sample a subset of descriptors to save memory
        if descriptors.shape[0] > 100:
            indices = np.random.choice(descriptors.shape[0], 100, replace=False)
            descriptors = descriptors[indices]
        
        # Add to the collection
        all_features.append(descriptors)
    
    # Combine all features
    all_features = np.vstack(all_features)
    
    # Ensure features are float32 (required by kmeans function)
    all_features = all_features.astype(np.float32)
    
    # Cap the number of features for clustering
    max_features_for_clustering = 10000
    if all_features.shape[0] > max_features_for_clustering:
        indices = np.random.choice(all_features.shape[0], max_features_for_clustering, replace=False)
        all_features = all_features[indices]
    
    # Perform k-means clustering to build vocabulary
    print(f"Clustering {all_features.shape[0]} features to create {vocab_size} centroids...")
    vocab = kmeans(all_features, num_centers=vocab_size)

    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    vocab_size = vocab.shape[0]
    
    # Process each image
    for img_path in tqdm(img_paths, desc="Creating Bag of SIFT features"):
        # Load and convert the image to grayscale
        img = Image.open(img_path).convert('L')
        img_array = np.array(img).astype('float32')
        
        # Extract SIFT features
        _, descriptors = dsift(img_array, step=[5, 5], fast=True)
        
        # If no features were found, create a uniform histogram
        if descriptors.shape[0] == 0:
            hist = np.ones(vocab_size) / vocab_size
            img_feats.append(hist)
            continue
            
        # Calculate distances between descriptors and vocabulary
        distances = cdist(descriptors, vocab)
        
        # Assign each descriptor to the nearest vocabulary word
        assignments = np.argmin(distances, axis=1)
        
        # Build histogram
        hist = np.zeros(vocab_size)
        for idx in assignments:
            hist[idx] += 1
            
        # Normalize histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        # Apply Hellinger kernel (take square root)
        hist = np.sqrt(hist)
        
        # L2 normalize the histogram
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
            
        img_feats.append(hist)
    
    # Convert to numpy array
    img_feats = np.array(img_feats)

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    test_predicts = []
    
    # Check if we're dealing with tiny images or bag of SIFT features
    is_bow = (train_img_feats.shape[1] > 100)
    
    if is_bow:
        # For bag of SIFT features, use Euclidean distance
        distances = cdist(test_img_feats, train_img_feats, metric='euclidean')
        
        # Choose k=15 for bag of SIFT (original parameter that achieved 0.6 accuracy)
        k = 15
    else:
        # For tiny images, use cosine distance which works well with normalized vectors
        distances = cdist(test_img_feats, train_img_feats, metric='cosine')
        
        # For tiny images, use k=7
        k = 7
    
    # Classify each test image
    for i in range(distances.shape[0]):
        # Find k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]
        
        # Get the labels of nearest neighbors
        neighbor_labels = [train_labels[idx] for idx in nearest_indices]
        
        # Count votes for each label
        label_counts = {}
        for label in neighbor_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        # Find the label with the most votes
        predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
        test_predicts.append(predicted_label)

    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts