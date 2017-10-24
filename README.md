# Bag of Visual Words

This repository contains a simple implementation of a descriptors to bag of words converter. The file contains functions on converting image descriptors to bag of words and training a k-NN to find the similar images based on bag of words.

## Functions

**init_SURF**

This function initializes the SURF object that will serve as the feature extractor of the image. The two parameters that this function accepts is first the hessianThreshold and the extended parameter.

* hessianThreshold - this is a threshold to decide from which value of you are willing to accept keypoints. The default value is 100
* extended - this parameter tells whether or not you want to use the normal SURF (64 pixels) or the extended SURF (128 pixels). Values can only be 0 or 1. 0 for normal SURF and 1 for extended SURF. The default value is 0

RETURNS the SURFObject

**retrieve_all_images**

This functions helps the user retrieve the features of all images in the directory. The strategy used by this function is it will iteratively load images by taking advantage of filenames that are named incrementally. (ex. cat.1.jpg, cat.2.jpg, etc.) There are five (5) parameters that are accepted by the function

* filepath - this is the filepath of the directory
* file_prefix - this is the text before the iteratable number of the files in the directory. If your filename is boy_face_1.jpg, then the file_prefix would be 'boy_face_'
* file_extension - the file extension of the files in the directory. Take note that the dot before the abbreviation must be included. (ex. '.jpg', '.png', etc.)
* num_images - this is the number of images you want to get the descriptors of. Make sure that num_images is the same as the number of images in the directory
* SURFObject - pass the SURFObject here

RETURNS image_names, image_descriptors

**create_patch_clusters**

This function creates the clusters of the patches generated from getting the descriptors of all the images you have in your directory. The parameters that this function takes are only two (2) namely: descriptors_of_images and num_clusters

* descriptors_of_images - these are the descriptors of all the images in your directory
* num_clusters - the number of clusters you want to have to represent your bag of words model.

RETURNS clusters 

**create_bag_of_visual_words** 

This function converts the a single image's descriptors into bag of words. This function takes in the following parameters

* descriptors - the image descriptors
* clusters - the KMeans clusters that are already trained

RETURNS image_bow

**convert_image_to_bow**

This function converts the a set of images' descriptors into bag of words. This function takes the following parameters:

* descriptors - the set of descriptors per image
* clusters - the KMeans clusters that were already trained

RETURNS image_bows

**scale_bows**

This simply normalizes the bows to prepare it for k-NN and determining the nearest neighbor. 

* image_bows - the image converted to bag of words will serve as the only parameter for this function

RETURNS sc, image_bows_normalized (returns the scaler and the normalized bows)

**train_knn**

Trains the k-NN for determining similar images. The parameters are as follows:

* image_bows - all the bag of words of the training images
* n_neighbors - the number of images you want to see. Default is equal to 5
* radius - default is equal to 1

RETURNS neighbors (the k-NN model)

**predict_similar_images**

This function determines the number of similar images based on the bag of words of the query image. The function has the following parameters

* image_bow - the bag of words of the image
* scaler - the normalization function for bow

RETURNS kneighbors along with the distances from the query image
