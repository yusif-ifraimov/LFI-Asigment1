## Yusif Ifraimov, MN: 915754
## Exercise 2: K-Means for color quantization (4 Punkte)
from typing import Any, Union

import numpy as np
import cv2
import sys
import random
import math
from sklearn.preprocessing import normalize ## for improved algorithm
from math import sqrt


############################################################
#
#                       KMEANS
#
############################################################

def distance(a, b):
    # implement distance metric - e.g. squared distances between pixels

    # YOUR CODE HERE:
    """
      usual representation of vectors here is as a row vector
      since np.dot looks at matching dimensions, 2nd vector must be transposed
    """
    return math.sqrt(np.dot(a-b, np.array(a-b).transpose()))

# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

# The code for k-Means is partially took/adapted from Machine Learning 1 course by Prof Timothy Downie, Beuth University of Applied Sciences.

def update_mean(img, clustermask, numclusters, centroids):
    ## Computing the new cluster center, i.e. numclusters mean colors


    new_centroids = np.zeros((numclusters, 1, 3), np.uint64) # list of num clusters RGB-vectors
    pixels_per_cluster = np.zeros(numclusters, np.uint32)

    # add all pixels within each cluster
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):            
            new_centroids[clustermask[i, j]][0] += img[i, j]
            pixels_per_cluster[clustermask[i, j]] += 1
            
    # find the mean of each cluster
    for i in range(numclusters):
        print(f"pixels in cluster {i}: {pixels_per_cluster[i]}")
        if pixels_per_cluster[i] > 0:
            centroids[i][0] = new_centroids[i][0] / pixels_per_cluster[i]     
        print(f"centroid in cluster {i}: {centroids[i]}")        
    print("\n")


def assign_centroid_to_pixel(pixel, centroids):
    """ Look for the closest centroid of a given pixel
    and save the index of that centroid in the clustermask 
    """
    
    # very large initialization of min. dist. to get started
    min_dist = sys.float_info.max
    index_centroid = 0
    
    # compute and (when needed) update min. distance
    for i in range(len(centroids)):
        curr_dist = distance(pixel, centroids[i][0])
        if curr_dist < min_dist:
            min_dist = curr_dist
            index_centroid = i
    
    return index_centroid
    

def assign_centroids_to_image(img, centroids, clustermask):
    """ Find the closest centroids for all the pixels in an image
    and save their indices in a given clustermask 
    (clustermask[i,j] gives the index of the cluster for the pixel at position [i,j]) 
    """
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            clustermask[i, j] = assign_centroid_to_pixel(img[i, j], centroids)
            
    

def assign_to_current_mean(img, result, clustermask, centroids, cluster_colors = None):
    """
    The function expects the img, the resulting image and a clustermask.
      After each call the pixels in result should contain a cluster_color corresponding to the cluster
      it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
      Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
      """
    ##  I. Displays the second image in which the clusters are highlighted: each pixel having a specific
    ##  cluster color given (these are not the actual colors, they are just representing the clusters!!) in cluster_colors
    ##  or take the color of the corresponding centroid
    ##  II. Then compute the overall error

    
    # initial error
    overall_dist = 0

    # YOUR CODE HERE
    # I. assign specific cluster color to each pixel in new image
    # II. and calculate the distance of every original pixel from its centroid
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # if no color palette is provided for the clusters, 
            # color them according to their centroids
            if cluster_colors is None:
                result[i, j] = centroids[clustermask[i, j]]
            else:
                result[i, j] = cluster_colors[clustermask[i, j]]
            overall_dist += distance(img[i, j], centroids[clustermask[i, j]][0])
    
    return overall_dist


def initialize(img, numclusters):
    """
    Initialize numcusters centroids at random
    """

    centroids = np.zeros((numclusters, 1, 3), np.uint64)
    
    # Choose random indices within the image
    randRows = random.sample(range(img.shape[0]), numclusters)
    randCols = random.sample(range(img.shape[1]), numclusters)
        
    for i in range(numclusters):
        centroids[i] = img[randRows[i], randCols[i]]
    
    return centroids

def initialize_improved(img, numclusters):
    """
    The exact algorithm is as follows:

    1.Choose one center uniformly at random among the data points.
    2.For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
    3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
    4. Repeat Steps 2 and 3 until k centers have been chosen.
    5. Now that the initial centers have been chosen, proceed using standard k-means clustering.
    """
    ## source: https://en.wikipedia.org/wiki/K-means%2B%2B

    centroids = np.zeros((numclusters, 1, 3), np.uint64)

    ## sample the data to RGB color space
    normalization = 255*sqrt(3)/(511*sqrt(2))
    # dimensions of image
    nRows = img.shape[0]*normalization
    nCols = img.shape[1]*normalization

    nRows = int(nRows)
    nCols = int(nCols)
    
    # choose random indices for the first cluster center
    randRow = random.sample(range(nRows), 1)
    randCol = random.sample(range(nCols), 1)
    centroids[0] = img[randRow, randCol]
    
    weights = []
    for w in range(numclusters-1):
        print(f"Iteration in optimized initialization: {w}\n")
        
        # calculate distances to centroids from all pixels
        for i in range(nRows):
            for j in range(nCols):
                indexCentroid = assign_centroid_to_pixel(img[i, j], centroids)
                weights.append(sqrt(distance(img[i, j], centroids[indexCentroid])**2 + (i*normalization)**2 + (j*normalization)**2) ** 2) ## measure the distance according to RGB color space

                 ## the code could be adapted to HSV and LAB color spaces as well,  by changing normalization parameter.
                 ## we can still use a euclidean distances for pixels instead of colors by removing the normalization parameter.
        
        # pick next centroid according to the weighted distribution
        maxIndex = nRows * nCols
        indexWrapped = random.choices(range(maxIndex), weights = weights, k = 1)
        # Note: random.choices() returns a list with one integer
        index_i = int(indexWrapped[0] / nRows) - 1
        index_j = indexWrapped[0] % nRows - 1
        centroids[w+1] = img[index_i, index_j]
        
        # remove current weights to add new ones in the next iteration
        weights.clear()
    
    return centroids


def kmeans(img, numclusters, cluster_colors = None, optimized = input('Enter True for optimized algorithm, or False for Main one:')):
    """ Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less than 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    
    # initializations of variables
    max_iter = 10
    min_change_rate = 0.02
    dist_old = sys.float_info.max
    it = 0 # counter for the iterations
    err = 1 # random initialization of current change rate
    clustermask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    result = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) # image of cluster colors
    
    # initialize centroids with kMeans++ or simple kMeans
    # Please keep in mind that running the code below with the optimized initialization may take up to 10 minutes!
    if optimized:
        centroids = initialize_improved(img, numclusters)
    else:
        centroids = initialize(img, numclusters)
        
    print(f"initial centroids: {centroids}")
    
    # iterate until max_inter is reached or error is relatively very small
    while it < max_iter and err > min_change_rate:
        print(f"iteration: {it}")
        
        # build the clustermask
        assign_centroids_to_image(img, centroids, clustermask)
        
        # calculate absolute error and build image of cluster colors
        dist_new = assign_to_current_mean(img, result, clustermask, centroids, cluster_colors)
        
        # relative error
        err = np.abs(dist_old - dist_new) / dist_old
        
        # move centroids to the new means of clusters
        update_mean(img, clustermask, numclusters, centroids)
        
        # update absolute error
        dist_old = dist_new
        it += 1
    print("\n\n")

    return result


"""
TESTING PART 
"""


# number of clusters
numclusters = 3
# corresponding colors for each cluster
cluster_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]]

# load the image and then scale it
imgRGB = cv2.imread('./Lenna.png')
scaling_factor = 0.5
imgRGB = cv2.resize(imgRGB, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# compare different color spaces and their result for clustering
imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
imgLAB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2LAB)

# execute k-means over the image in different color spaces
print(f"KMeans for RGB\n")
resRGB = kmeans(imgRGB, numclusters, cluster_colors)
print(f"KMeans for HSV\n")
resHSV = kmeans(imgHSV, numclusters, cluster_colors)
print(f"KMeans for LAB\n")
resLAB = kmeans(imgLAB, numclusters, cluster_colors)

# dimensions of one image
h, w = resRGB.shape[:2]

# define window
vis = np.zeros((3 * h, 4 * w, 3), np.uint8)
vis[ : h, :w] = resRGB
vis[ : h, w : 2*w] = imgRGB
vis[h :2*h, :w] = resHSV
vis[h:2*h, w : 2*w] = imgHSV
vis[2*h:3*h, :w] = resLAB
vis[2*h:3*h, w : 2*w] = imgLAB

# The approach as in above part, but for 6 clusters
numclusters = 6
print(f"KMeans for RGB\n")
resRGB = kmeans(imgRGB, numclusters, cluster_colors)
print(f"KMeans for HSV\n")
resHSV = kmeans(imgHSV, numclusters, cluster_colors)
print(f"KMeans for LAB\n")
resLAB = kmeans(imgLAB, numclusters, cluster_colors)

vis[ : h, 2*w:3*w] = resRGB
vis[ : h, 3*w : 4*w] = imgRGB
vis[h :2*h, 2*w:3*w] = resHSV
vis[h:2*h, 3*w : 4*w] = imgHSV
vis[2*h:3*h, 2*w:3*w] = resLAB
vis[2*h:3*h, 3*w : 4*w] = imgLAB

# compare kmeans for different number of clusters
# by assigning each cluster the color of its centroid
k = [4, 16, 32, 64]
stripe = np.zeros((h, 5 * w, 3), np.uint8)
stripe[ : h, : w] = imgRGB
for i in range(len(k)):
    imgKMeans = kmeans(imgRGB,  k[i])
    stripe[ : h, (i+1)*w : (i+2)*w] = imgKMeans

cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
cv2.imshow("Comparison for different number of clusters", stripe)
cv2.waitKey(0)
cv2.destroyAllWindows()
