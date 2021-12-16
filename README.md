# AIClassFacialProject
AI Class Project

The purpose of this project is to map out and define individual facial features of the human face. It is building off the work created by https://github.com/italojs/facial-landmarks-recognition italojs, and expanding on it.
After we use his program to landmark a face, we take measurements based on the placing of those landmarks and then plot them, before running k-means clustering to give us distinct categories of each facial feature of a range of sizes or distances.


Project Outline:
First, we run the faces of the dataset through landmarks detector to make sure each image we're about to use has two eyes and only one face. Next we use align_all.py to make sure each photo is greyscaled, cropped, and ajusted left or right around the center point between the eyese. This ensures that each face is oriented the same way when we run them through Picture.py. As they go through the model, it scans their faces, places the landmarks, then stores the coordinates into an array. The array is then plotted and then fed to kmeans where we get the clusters of facial features for necessary categories. 
