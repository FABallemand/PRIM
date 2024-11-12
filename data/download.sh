#!/bin/bash

# Download vimeo dataset
mkdir vimeo
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
mv vimeo_triplet.zip vimeo/vimeo_triplet.zip
unzip vimeo/vimeo_triplet.zip -d vimeo

# Download OpenImage dataset
# mkdir open_images
# Use TensorFlow Datasets

# Download Kodak dataset
mkdir kodak
# Download manually on Kaggle

# Download CLIC dataset
mkdir clic
mkdir clic/validation
mkdir clic/test
wget https://storage.googleapis.com/clic2023_public/validation_sets/clic2024_validation_image.zip
mv clic2024_validation_image.zip clic/clic2024_validation_image.zip
unzip clic/clic2024_validation_image.zip -d clic/validation
wget https://storage.googleapis.com/clic2023_public/test_sets/clic2024_test_image.zip
mv clic2024_test_image.zip clic/clic2024_test_image.zip
unzip clic/clic2024_test_image.zip -d clic/test