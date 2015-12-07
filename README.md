Corey Grief and Adam Snyder  
Bryan Pardo  
EECS 439 Machine Learning  
3 December 2015  
Homework 9  

**Part 1**

A. total: 60,000 0: 5,923 1: 6,742 2: 5,958 3: 6,131 4: 5,842 5: 5,421 6: 5,918 7: 6,265 8: 5,851 9: 5,949

B. We looked through examples of the number 7 and found these to be shocking.  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/bad_7_a.png)  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/bad_7_b.png)  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/bad_7_c.png)  
These examples might be challenging because they appear to be other numbers, like 9 or some versions of 1.

C. We chose our training and testing sets randomly without replacement. For a classifier to generalize over unseen data,
we want the training and testing sets to be representative of the population of interest.In a perfect world, we would 
look through all of the data in order to get examples that both cover all of the variations in the examples,
and are representative of the data as a whole. However, this in infeasible due to the size of the data set.
Therefore, we chose to randomly draw from the examples without replacement in order to get a set that is approximately
representative. Our training and testing sets are disjoint, and training set is 1000 of each digit, for 10000 total 
images. The testing set is 100 of each digit for 1000 total images.

D. The images are converted into a vector for classification where each pixel is a feature. It is important that 
similarly labeled examples line up so that their feature vectors appear similar. Because of the way the data set is
structured, the classifier is actually learning which pixels are on or off and the correlation between pixels over the 
examples.

**Part 2**

Naive Bayes

A. Naive Bayes Classifiers are simple probability-based classifiers that apply Bayes' theorem with the assumption of
independence between features given the class variable. The classifiers consider each of the features to contribute
independently to the probability of an example falling within a given class. This assumption causes such classifiers to
be quite simple and efficient and only requires a small amount of training data.

Features on the digit data for this classifier would be each pixel in the image, which can take a value between 0 or 1
depending on that pixel's intensity. If the pixel has an intensity greater than .5, it is 1, otherwise it is 0.
This kind of data obviously violates the independence assumption of the classifier, but naive bayes often performs well
even when this assumption is violated. The classifier will output a class corresponding to the digit that the 
classifier classified the input as.

B. Hyperparameters for Bernoulli naive bayes:
alpha: Additive smoothing parameter. This attempts to smooth categorical data by adding a certain baseline to each
feature, such that even features that are not observed have some small probability. (e.g. black swan discussion).

* binarize: threshold for converting features to booleans. Allows for changing what the pixel intensity threshold is for 
marking it as "on" vs "off"
* fit_prior: whether to learn class prior probabilities or assume a uniform prior probability
* class_prior: manual input for class prior probabilities.

Support Vector Classification

A. A support vector machine creates a binary classifier by mapping example inputs into a higher dimensional space using 
a kernel function and then finding the hyperplane that separates the examples into two groups such that the margin 
between the groups is maximized. To use it to classify images, the data would be encoded as a list of binary digits 
representing each pixel in the image as being either on or off. The SVM would interpret each pixel as a dimension. The 
classifier will output the class of the predicted value of the test image.

B. Hyperparameters for SVC:
* kernel: the kernel function to use, which changes how new training data is mapped to a higher dimension space
* C: the cost of classification. A large C leads to low bias and high variance, while a small C leads to high bias and 
low variance.
* gamma: kernel coefficient, which controls the sharpness of the peaks where the points are raised. A high gamma gives 
a soft bump, resulting in high bias and low variance, while a low gamma gives a sharp bump, resulting in low bias and 
high variance.


**Part 3**

A.

B.

C.

**Part 4**

B.  
kernel='rbf', C=1, gamma='auto' :
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_set_size.png)

training_set_size=10000, degree=3, C=1, gamma='auto' :  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_kernel.png)

training_set_size=10000, kernel='rbf', gamma='auto' :  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_c.png)

training_set_size=10000, kernel='rbf', C=8 :  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_gamma.png)  

Based on my results, the default RBF kernel function performs the best. Error is further minimized by increasing the C 
value, which corresponds to the cost of classification. A large C leads to low bias and high variance, while a small C 
leads to high bias and low variance. Finally, the gamma value can be tweaked to optimize accuracy, with a value on the 
order of 0.1 being most optimal. Gamma affects how much influence each training example has.

After a performing grid search to optimize the parameters further, the best SVM classifier had hyperparameters 
kernel=**'rbf'**, C=**8**, and gamma=**0.02**, and run with a training_set_size=**10000** had error=**0.029**.

Confusion Matrix:  
```
     0  1  2  3  4  5  6  7  8  9  
  -------------------------------  
0 | 99  0  0  0  0  0  1  0  0  0  
1 |  0 99  0  0  0  0  0  1  0  0  
2 |  3  1 92  0  2  1  0  1  0  0  
3 |  0  0  0 98  0  0  0  1  0  1  
4 |  0  1  0  0 98  0  0  0  0  1  
5 |  1  0  0  2  0 96  0  0  0  1  
6 |  0  2  0  0  0  0 98  0  0  0  
7 |  1  0  0  0  0  0  0 99  0  0  
8 |  0  1  1  1  0  0  0  0 97  0  
9 |  0  0  1  2  2  0  0  0  0 95
```

C.  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_miss_0.png)  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_miss_1.png)  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_miss_7.png)  
![](https://github.com/friendly-flame/codename-obtuse-sniffle/blob/master/images/svm_miss_9.png)  
The misclassified examples were a mix of poor examples, clean examples, and examples that were predicted to fail from 
part 1B. As such, we have learned that not all of the examples are perfect, but since the SVM treats each pixel as 
a feature instead of looking at aspects of the overall shape of the image as features, images like 2, which shares 
many pixels with the 0, are misclassified. The fact that there is a gap in the stroke is not factored into the 
classification.

**Part 5**

**Part 6**

A.

B.

C.
