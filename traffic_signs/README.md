# Traffic Sign Recognition Program

## Writeup - Daniel Alejandro Reyna Torres

The goal is to write a software pipeline to identify and recognise traffic signs.

---

## The Project

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

## Data Set Summary & Exploration

The very fist step in every Machine Learning task is to load and understand the data. The data set corresponds to traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). I used the numpy and pandas libraries to calculate summary statistics of the traffic signs data set:


Above is an **exploratory visualization** of the training set. Summary of data is:

- Number of training examples = 34799
- Number of testing examples = 12630
- Number of validating examples = 4410
- Image data shape = (32, 32, 3)
- Number of classes = 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the amount of samples for each traffic sign is distributed. 

![dataset_dist]

If we take a closer look on the data set we can see the following: 

![class_freq]

Traffic signs with most samples:
- Speed limit (50km/h) - 2010 samples
- Speed limit (30km/h) - 1980 samples
- Yield - 1920 samples
- Priority Road - 1890 samples
- Keep Right - 1860 samples

Traffic signs with fewer samples:
- Speed limit (20km/h) - 180 samples
- Dangerous curve to the left - 180 samples
- Go straight or left - 180 samples
- Pedestrians - 210 samples
- End of all speed and passing limits - 210 samples

It can be seen that there is an uneven number of samples for each traffic sign. Between the sign with most samples and the one with less samples, there are **1830** samples! This is something to consider in the design of the classification pipeline since this class imbalance could bring wrong classification results because the model would be reflecting the underlying class distribution.

Here are some samples from the data set.
![dataset]

Now, let's deep dive into our pipeline for traffic sign classification!

---

## Design and Test a Model Architecture


### Model Architecture

Training with coloured images and using the original LeNet-5 architecture introduced by [LeCun et al. in their 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). This yield to an accuracy of 85% without any changes, seems to be a good start. Let's try to improve it.

Architecture of LeNel-5:
![lenet]

### Model Training

Training was donde in keras with the following hyperparameters, reaching an accuracy of XX% were:

- Batch size: 32
- Epochs: 5
- Learning rate: 0.001
- Dropout rate: 0.5
- Color channels: 3

My final model consists of the following architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 COLOURED image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x8 	|
| RELU					| Activation (Nonlinearity)						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					| Activation (Nonlinearity)						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               |Flatten the output shape 3D->1D                |
| Fully connected		| Array of 120 elements        					|
| RELU					| Activation (Nonlinearity)						|
| Dropout	    		| PENDING Regularisation								|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| Array of 84 elements        					|
| RELU					| Activation (Nonlinearity)						|
| Dropout	    		| PENDING Regularisation								|
| Fully connected		| Array of 43 elements (number of classes) 		|
| Softmax				| Probabilities for each predicted class   		|

### Solution Approach

As mentioned, the original LeNet-5 architecture was a very good start in order to classify traffic signs. Based on the LeNet-5, my final solution included the Adam optimizer (already implemented in the LeNet lab), added two [droput](https://arxiv.org/pdf/1207.0580.pdf) layers between the fully connected layers, a color-to-grayscale process and normalisation.

Final results (so far):

- Valid Accuracy: 85.01%
- Test Accuracy: 58.00%


![UPDATE THIS results] UPDATE THIS

## Test a Model on New Images

Here are some German traffic signs that I found on the web:

![german_signs]

Here are the results of the prediction:

| Image			            |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Keep left      		    | Keep left   									| 
| No entry     			    | No entry 										|
| Continue				    | Continue										|
| Stop	      		        | No passing					 				|
| Speed limit (70km/h)		| Speed limit (70km/h)      					|
| Turn left ahead			| Turn left ahead      							|
| Children crossing			| Children crossing      						|
| Road work			| Road work      							            |

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. For the first image, the model is 99.37% sure that this is a keep left! The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9937         		| Keep left   									| 
| .0063     			| Speed limit (120km/h) 						|
| .0					| Keep right									|
| .0	      			| Speed limit (30km/h)			 				|
| .0				    | Yield      					        		|

All images were in the same way analysed through the classification pipeline. For each row in the next image: the first image is the original input image and the rest are the top five softmax predictions.



## Final Results


---

## Discussion


--

_Daniel_


[class_freq]: report_images/Class_Freq.png
[dataset_dist]: report_images/Traffic_Signs_Distribution.png
[dataset]: report_images/Explore_DS.png
[gray_normalisation]: report_images/normalisation.png
[minmax]: report_images/minmaxnorm.png
[lenet]: report_images/lenet.png
[results]: report_images/Valid_accuracy.png
[german_signs]: report_images/New_Images.png
[german_signs_pred]: report_images/New_Img_Pred.png
