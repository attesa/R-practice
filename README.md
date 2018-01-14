R-practice

Small R based data science projects, inlcuding regression, classification, random forest, SVM, neural network models 

Practice 1

1) (10 points) (Exercise 9 modified, ISL) In this exercise, we will predict the number
of applications received using the other variables in the College data set in the ISLR
package.
(a) Split the data set into a training set and a test set. Fit a linear model using least
squares on the training set, and report the test error obtained.
(b) Fit a ridge regression model on the training set, with λ chosen by crossvalidation.
Report the test error obtained.
(d) Fit a lasso model on the training set, with λ chosen by crossvalidation.
Report the test error obtained, along with the number of non-zero coefficient
estimates.
(e) Fit a PCR model on the training set, with k chosen by cross-validation. Report the
test error obtained, along with the value of k selected by cross-validation.
(f) Fit a PLS model on the training set, with k chosen by crossvalidation.
Report the test error obtained, along with the value of k selected by cross-validation.
(g) Comment on the results obtained. How accurately can we predict the number of
college applications received? Is there much difference among the test errors resulting
from these five approaches?
2) (10 points) The insurance company benchmark data set gives information on
customers. Specifically, it contains 86 variables on product-usage data and sociodemographic
data derived from zip area codes. There are 5,822 customers in the
training set and another 4,000 in the test set. The data were collected to answer
the following questions: Can you predict who will be interested in buying a
caravan insurance policy and give an explanation why? Compute the OLS
estimates and compare them with those obtained from the following variableselection
algorithms: Forwards Selection, Backwards Selection, Lasso regression,
and Ridge regression. Support your answer.
(The data can be downloaded from https://kdd.ics.uci.edu/databases/tic/tic.html. )
3) (10 points) (Exercise 9 modified, ISL) We have seen that as the number of
features used in a model increases, the training error will necessarily decrease, but
the test error may not. We will now explore this in a simulated data set.
Generate a data set with p = 20 features, n = 1, 000 observations, and an
associated quantitative response vector generated according to the model
𝑌 = 𝑋𝛽 + 𝜀
where β has some elements that are exactly equal to zero. Split your data set into
a training set containing 100 observations and a test set containing 900
observations.
Perform best subset selection on the training set, and plot the training set MSE
associated with the best model of each size. Plot the test set MSE associated with
the best model of each size.
For which model size does the test set MSE take on its minimum value?
Comment on your results. How does the model at which the test set MSE is
minimized compare to the true model used to generate the data? Comment on the
coefficient values.



Practice 2

1) (10 points) Using the Boston data set (ISLR package), fit classification models in
order to predict whether a given suburb has a crime rate above or below the
median. Explore logistic regression, LDA and kNN models using various subsets
of the predictors. Describe your findings.
2) (10 points) Download the diabetes data set
(http://astro.temple.edu/~alan/DiabetesAndrews36_1.txt). Disregard the first
three columns. The fourth column is the observation number, and the next five
columns are the variables (glucose.area, insulin.area, SSPG, relative.weight,
and fasting.plasma.glucose). The final column is the class number. Assume the
population prior probabilities are estimated using the relative frequencies of the
classes in the data.
(Note: this data can also be found in the MMST library)
(a) Produce pairwise scatterplots for all five variables, with different symbols or
colors representing the three different classes. Do you see any evidence that
the classes may have difference covariance matrices? That they may not be
multivariate normal?
(b) Apply linear discriminant analysis (LDA) and quadratic discriminant analysis
(QDA). How does the performance of QDA compare to that of LDA in this
case?
(c) Suppose an individual has (glucose area = 0.98, insulin area =122, SSPG =
544. Relative weight = 186, fasting plasma glucose = 184). To which class
does LDA assign this individual? To which class does QDA?
3) a) Under the assumptions in the logistic regression model, the sum of posterior
probabilities of classes is equal to one. Show that this holds for k=K.
b) Using a little bit of algebra, show that the logistic function representation and
the logit representation for the logistic regression model are equivalent.
In other words, show that the logistic function:
𝑝 𝑋 =
exp (𝛽! + 𝛽!𝑋)
1 + exp (𝛽! + 𝛽!𝑋)
is equivalent to:
𝑝(𝑋)
1 − 𝑝(𝑋)
= exp 𝛽! + 𝛽!𝑋 .
4) (10 points) We will now perform cross-validation on a simulated data set.
Generate simulated data as follows:
> set.seed(1)
>x=rnorm(100)
>y=x-2*x^2+rnorm(100)
a) Compute the LOOCV errors that result from fitting the following four models
using least squares:
Y = β! + β!X + ε
Y = β! + β!X + β!X! + ε
Y = β! + β!X + β!X! + β!X! + ε
Y = β! + β!X + β!X! + β!X! + β!X! + ε
a) Which of the models had the smallest LOOCV error? Is this what you
expected? Explain your answer.
b) Comment on the statistical significance of the coefficient estimates that results
from fitting each of the models in part c using least squares. Do these results
agree with the conclusions drawn from the cross-validation?
5) (10 points) When the number of features (p) is large, there tends to be a deterioration
in the performance of KNN and other local approaches that perform prediction using only
observations that are near the test observation for which a prediction must be made. This
phenomenon is known as the curse of dimensionality, and it ties into the fact that
non-parametric approaches often perform poorly when p is large.
a) Suppose that we have a set of observations, each with measurements on p = 1
feature, X. We assume that X is uniformly (evenly) distributed on [0, 1].
Associated with each observation is a response value. Suppose that we wish to
predict a test observation’s response using only observations that are within 10%
of the range of X closest to that test observation. For instance, in order to predict
the response for a test observation with X = 0. 6, we will use observations in the
range [0. 55, 0. 65]. On average, what fraction of the available observations will
we use to make the prediction?
b) Now suppose that we have a set of observations, each with measurements on p =
2 features, X1 and X2 . We assume that (X1, X2) are uniformly distributed on [0,
1] x [0, 1]. We wish to predict a test observation’s response using on observations
that are within 10% of the range of X1 and within 10% of the range of X2 closest
to that test observation. For instance, in order to predict the response for a test
observation with X1 = 0. 6 and X2 = 0. 35, we will use observations in the range
[0. 55, 0. 65] for X1 and in the range [0. 3, 0. 4] for X2. On average, what fraction
of the available observations will we use to make the prediction?
c) Now suppose that we have a set of observations on p = 100 features. Again the
observations are uniformly distributed on each feature, and again each feature
ranges in value from 0 to 1. We wish to predict a test observation’s response using
observations within the 10% of each feature’s range that is closest to that test
observation. What fraction of the available observations will we use to make the
prediction?
d) Use your answers from (a-c) to argue the drawback of KNN when p is large.










Practice 3
1. (10 points) (Exercise 7.9) For the prostate data of Chapter 3, carry out a bestsubset
linear regression analysis, as in Table 3.3 (third column from the left).
Compute the AIC, BIC, five- and tenfold cross-validation, and bootstrap .632
estimates of prediction error.
2) (10 points) A access the wine data from the UCI machine learning repository
(https://archive.ics.uci.edu/ml/datasets/wine). These data are the results of a
chemical analysis of 178 wines grown over the decade 1970-1979 in the same
region of Italy, but derived from three different cultivars (Barolo, Grignolino,
Barbera). The Babera wines were predominately from a period that was much
later than that of the Barolo and Grignolino wines. The analysis determined the
quantities MalicAcid, Ash, AlcAsh, Mg, Phenols, Proa, Color, Hue, OD, and
Proline. There are 50 Barolo wines, 71 Grignolino wines, and 48 Barbera wines.
Construct the appropriate-size classification tree for this dataset. How many
training and testing samples fall into each node? Describe the resulting tree and
your approach.
3) (10 points) Apply bagging, boosting, and random forests to a data set of your
choice (not one used in the committee machines labs). Fit the models on a
training set, and evaluate them on a test set. How accurate are these results
compared to more simplistic (non-ensemble) methods (e.g., logistic regression,
kNN, etc)? What are some advantages (and disadvantages) do committee
machines have related to the data set that you selected?
4) (10 points ~ Exercise 15.6) Fit a series of random-forest classifiers to the SPAM
data, to explore the sensitivity to m (the number of randomly selected inputs for
each tree). Plot both the OOB error as well as the test error against a suitably
chosen range of values for m.
(5) (10 points; Exercise 11.7) Fit a neural network to the spam data of Section 9.1.2.
The data is available through the package “ElemStatLearn”. Use cross-validation
or the hold out method to determine the number of neurons to use in the layer.
Compare your results to those for the additive model given in the chapter. When
making the comparison, consider both the classification performance and
interpretability of the final model.
(6) (10 points) Take any classification data set and divide it up into a learning set and
an independent test set. Change the value of one observation on one input
variable in the learning set so that the value is now a univariate outlier. Fit
separate single-hidden-layer neural networks to the original learning-set data and
to the learning-set data with the outlier. Use cross-validation or the hold out
method to determine the number of neurons to use in the layer. Comment on the
effect of the outlier on the fit and on its effect on classifying the test set. Shrink
the value of that outlier toward its original value and evaluate when the effect of
the outlier on the fit vanishes. How far away must the outlier move from its
original value that significant changes to the network coefficient estimates occur?
(7) (10 points; ISLR modified Ch9ex8) This problem involves the OJ data set in the
ISLR package. We are interested in the prediction of “Purchase”. Divide the data
into test and training.
(A) Fit a support vector classifier with varying cost parameters over the range
[0.01, 10]. Plot the training and test error across this spectrum of cost parameters,
and determine the optimal cost.
(B) Repeat the exercise in (A) for a support vector machine with a radial kernel.
(Use the default parameter for gamma). Repeat the exercise again for a support
vector machine with a polynomial kernel of degree=2. Reflect on the
performance of the SVM with different kernels, and the support vector classifier,
i.e., SVM with a linear kernel.
