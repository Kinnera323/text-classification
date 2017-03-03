#!/bin/bash

echo " Welcome to Text Classification "

echo "Installing dependencies"

sudo apt-get install python-pip
pip install nltk
pip install --upgrade pip
sudo easy_install pip
sudo pip install -U numpy
sudo pip install -U pyyaml nltk
sudo python -m nltk.downloader reuters
sudo python -m nltk.downloader stopwords
sudo python -m nltk.downloader punkt

echo "cloning Code"

git clone git clone https://github.com/Kinnera323/text-classification.git

cd text-classification/

echo "Processing the file"
python preprocessing.py

echo "caluclating Frequent patterns and rules"

python Apriori/apriori.py -f output.csv -s 0.17 -c 0.68

echo "end of script"

