# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import matplotlib as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# This function initializes the SURF object that will serve as the
# feature extractor of the image. The two parameters that this function
# accepts is first the hessianThreshold and the extended parameter.
# 
# PARAMETERS: 
# hessianThreshold - this is a threshold to decide from which value of you are willing to accept keypoints. The default value is 100
# extended - this parameter tells whether or not you want to use the normal SURF (64 pixels) or the extended SURF (128 pixels). Values can only be 0 or 1. 0 for normal SURF and 1 for extended SURF. The default value is 0
def init_SURF(hessianThreshold=100, extended=0):
    SURFObject = cv2.SURF(hessianThreshold=hessianThreshold, extended=extended)
	# retunrs a SURF object
    return SURFObject    

# This functions helps the user retrieve the features of all images in 
# the directory. The strategy used by this function is it will 
# iteratively load images by taking advantage of filenames that are 
# named incrementally. (ex. cat.1.jpg, cat.2.jpg, etc.) There are five 
# (5) parameters that are accepted by the function
#
# PARAMETERS:
# filepath - this is the filepath of the directory
# file_prefix - this is the text before the iteratable number of the files in the directory. If your filename is boy_face_1.jpg, then the file_prefix would be 'boy_face_'
# file_extension - the file extension of the files in the directory. Take note that the dot before the abbreviation must be included. (ex. '.jpg', '.png', etc.)
# num_images - this is the number of images you want to get the descriptors of. Make sure that num_images is the same as the number of images in the directory
# SURFObject - pass the SURFObject here
def retrieve_all_images(filepath, file_prefix, file_extension, num_images, SURFObject):
    image_names = []
    image_descriptors = []
    
    for i in range(1, num_images):
        filename = file_prefix + str(i) + file_extension
        print("Image Number: " + str(i) + " " + filename)
        img = cv2.imread(filepath + filename)
        keypoints, descriptors = SURFObject.detectAndCompute(img, None)
        image_descriptors.append(descriptors)
        image_names.append(filename)
    
	# returns the images names and the descriptors of the images in the directory
    return image_names, image_descriptors

	
# This function creates the clusters of the patches generated from 
# getting the descriptors of all the images you have in your directory. 
# The parameters that this function takes are only two (2) namely: 
# descriptors_of_images and num_clusters
#
# PARAMETERS:
# descriptors_of_images - these are the descriptors of all the images in your directory
# num_clusters - the number of clusters you want to have to represent your bag of words model
def create_patch_clusters(descriptors_of_images, num_clusters):  
    clusters = KMeans(n_clusters = num_clusters, init="k-means++")
    clusters.fit(np.vstack(descriptors_of_images))
	# returns a trained KMeans cluster
    return clusters

# This function converts the a single image's descriptors into bag of 
# words. This function takes in the following parameters
#
# PARAMETERS:
# descriptors - the image descriptors
# clusters - the KMeans clusters that are already trained
def create_bag_of_visual_words(descriptors, clusters):
    image_bow = np.zeros(clusters.get_params()['n_clusters'])
    for i in range(0, len(descriptors)):
        cluster = clusters.predict(descriptors[i].reshape(1,-1))
        image_bow[cluster] = image_bow[cluster] + 1
	# returns the bag of words representation of a single image
    return image_bow

# This function converts the a set of images' descriptors into bag of words. 
# This function takes the following parameters:
# 
# PARAMETERS: 
# descriptors - the set of descriptors per image
# clusters - the KMeans clusters that were already trained
def convert_images_to_bows(descriptors, clusters):
    image_bows = []
    for i in range(0, len(image_descriptors)):
        image_bows.append(create_bag_of_visual_words(image_descriptors[i], clusters))
    image_bows = np.vstack(image_bows)
	# returns the bag of words of each image 
    return image_bows

	
# This simply normalizes the bows to prepare it for k-NN 
# and determining the nearest neighbor.
# 
# PARAMETERS:
# image_bows - the image converted to bag of words will serve as the only parameter for this function
def scale_bows(image_bows):
    sc = StandardScaler()
    image_bows_normalized = sc.fit_transform(image_bows)
	# returns a StandardScaler which can be used to normalize query image (represented by bag of words)
	# and also returns the normalized bag of words of all images
    return sc, image_bows_normalized

# Trains the k-NN for determining similar images. The parameters are as follows:
# 
# PARAMETERS:
# image_bows - all the bag of words of the training images
# n_neighbors - the number of images you want to see. Default is equal to 5
# radius - default is equal to 1
def train_knn(image_bows, n_neighbors=5, radius=1):
    neighbors = NearestNeighbors(n_neighbors = n_neighbors, radius = radius, n_jobs = 2)
    neighbors.fit(image_bows)
	# returns the k-NN model
    return neighbors

# This function determines the number of similar images 
# based on the bag of words of the query image. The 
# function has the following parameters
#
# PARAMETERS:
# image_bow - the bag of words of the image
# scaler - the normalization function for bow
def predict_similar_images(image_bow, scaler):
    image_bow = scaler.transform(image_bow.reshape(1,-1))
	# returns the similar images indices along with their distances from the bag of words of the similar images.
    return neighbors.kneighbors(image_bow.reshape(1,-1))

##
##     EXAMPLE USAGE:
##	
	
# specify the filepath
filepath1 = '\\cats\\'
filepath2 = '\\dogs\\'

# initialize the feature descriptor
SURFObject = init_SURF(400, 0)

# get the descriptors of all the images in the directory
image_names1, image_descriptors1 = retrieve_all_images(filepath1, "cat.",".jpg",1001,SURFObject)
image_names2, image_descriptors2 = retrieve_all_images(filepath2, "dog.",".jpg",1001,SURFObject)

# append the lists to each other
image_names = image_names1 + image_names2
image_descriptors = image_descriptors1 + image_descriptors2

# generate the clusters of the patches
clusters = create_patch_clusters(np.vstack(image_descriptors), num_clusters=100)

# convert images into bage of visual words
image_bows = convert_images_to_bows(image_descriptors, clusters)

# normalize the bag of words for k-nearest neighbor
scaler, normalized_bows = scale_bows(image_bows)

# train the k-NN
neighbors = train_knn(normalized_bows, 5, 1)

# determine similar images 
print(predict_similar_images(image_bows[0],scaler))
    







