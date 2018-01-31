# GD-Link
This is the MATLAB code for GD-Link.GD-Link is to link accounts of the same user across different social platforms.

This example is executed in the data set Facebook-Twitter. There are around 3600 link pairs cross two platform, and the original data has been preprocessed to user feature vector.

Because CVX or MOSEK optimization method as mentioned in the paper is related to the third-party software, the gradient descent is used as optimization method in this version. More optimization algorithms will be available in following versions.
