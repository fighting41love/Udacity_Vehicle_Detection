**Vehicle Detection Project**
The goals / steps of this project are the following:
> Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

> Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

> Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

> Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.


### Histogram of Oriented Gradients (HOG)
#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code for this step is contained in the `Cell 14` of the IPython notebook . I started by reading in all the vehicle and non-vehicle images in the `Cell 13`. Here is the examples  of the vehicle and non-vehicle classes:

![Vehicle and non-vehicle images](http://upload-images.jianshu.io/upload_images/2528310-a86f74bee987db90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

I then explored different color spaces and different skimage.hog()
 parameters .
```
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
```
I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.
Here is an example using the YCrCb color space:
![Hog](http://upload-images.jianshu.io/upload_images/2528310-5d5457140c146f03.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

####2. Explain how you settled on your final choice of HOG parameters.
I tried various combinations of parameters. The `YCrCb` is the best choice. Others can also use the `GRAY` colors pace. However, the `GRAY` which neglect the color information may loose the experimental result. I finally set the parameters in `Cell 17` as follows:
```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
visualize = True # Visualize hog image on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
scale = 1.5 # A parameter for the function finding cars
```

####3. Describe how you trained a classifier using your selected HOG features (and color features if you used them).
In `Cell 17`, I trained a linear SVM using the normalized HOG features. The features are first normalized as follows:

```
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
# Compute the mean and std to be used for later scaling.
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
# Perform standardization by centering and scaling
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
```
Then, the overall dataset is split as training and test data. 
```
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
```
A linear svm model is employed to fit the training data. 
```
svc = LinearSVC()
svc.fit(X_train, y_train)
```
We predict the labels of the test samples as follows:
```
svc.predict(X_test[0:n_predict])
```
The performance of the model can be evaluated as follows:
```
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
```
### Sliding Window Search
#### 1. Describe how you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
It's not a good idea to search random window all over the image. I decided to search random window positions at random scales just at the bottom of the image like this: 

![Windows for vehicles detection](http://upload-images.jianshu.io/upload_images/2528310-cc5502a8116f1874.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

I plot the heat map of the windows.

![Heat map](http://upload-images.jianshu.io/upload_images/2528310-76a5d30724b306f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

We have a false positive in the left part of the image. It is not a car. We try to remove the false positive using a threshold to remove the single window. 
```
def apply_threshold(heatmap, threshold): # Zero out pixels below the threshold in the heatmap
    heatmap[heatmap < threshold] = 0 
    return heatmap 
heated = apply_threshold(heat_b,3)
``` 

![Filtered heat map](http://upload-images.jianshu.io/upload_images/2528310-17595c3ecac7909b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

Finally, we combine the detected windows  with the previous image from camera. 


![Frame with windows](http://upload-images.jianshu.io/upload_images/2528310-b52d1cf9e3ae901c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?
Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![test 1](http://upload-images.jianshu.io/upload_images/2528310-2e229838cd6b1b21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

![test 2](http://upload-images.jianshu.io/upload_images/2528310-07a57ec28d1a8bbc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video. Here's a [link to my video result](https://github.com/fighting41love/Udacity_Vehicle_Detection/blob/master/project_video_output.mp4).
We also upload the video to [youtube]().

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
I illustrate how we solve the problem in `Sliding Window Search` part. To combine the overlapping bounding boxes, we first use the min-max function to generate the boxes. Sometimes the box is too large. Hence, we write a detection class to average the box boundary.

### Discussion
#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

> The boxes in the image is not solid. The box in a new frame may jump too far away from the previous frame. We smooth the boxes' position by averaging the positions in the past 10 frames.

> I wonder whether we should identify the vehicles from the opposite direction. My codes sometimes can detect vehicles from the opposite direction. However, sometimes it doesn't. 

> Some parameters in our codes are fixed. I don't know the model will work on some extreme weather conditions.
