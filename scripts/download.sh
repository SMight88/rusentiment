#!/usr/bin/env bash

# Download and unzip of VK embeddings
cd ../data
mkdir embeddings
cd embeddings
wget http://text-machine.cs.uml.edu/lab2/data/fasttext.min_count_100.vk_posts_all_443550246.300d.vec.zip
unzip fasttext.min_count_100.vk_posts_all_443550246.300d.vec.zip

# Download dataset
cd ../
mkdir dataset
cd dataset
wget https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_preselected_posts.csv
wget https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_random_posts.csv
wget https://raw.githubusercontent.com/text-machine-lab/rusentiment/master/Dataset/rusentiment_test.csv



