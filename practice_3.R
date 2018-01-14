


library(ElemStatLearn)
library(glmnet)
library(pls)
library(leaps)
#1#

set.seed(5509)
#read data
pro_1_train = sample(1:nrow(prostate), nrow(prostate)*0.80)
pro_1_test = -pro_1_train
pro_1_train_data = prostate[pro_1_train, ]
pro_1_test_data = prostate[pro_1_test, ]
best_sub = regsubsets(train~.,data = prostate,method="exhaustive")
reg_summary=summary(best_sub)
reg_summary$cp
#0.4650382 1.3504137 0.4181619 1.3700022 2.3234078 4.1744943 6.1001435 8.0006358
#third is the best
reg_summary$bic
#6.040028  9.451005 10.895878 14.326838 17.745890 22.155045 26.646990 31.110819
#first is the best

#5-fold
best_sub = regsubsets(train~.,data = pro_1_train_data,method="exhaustive")
train_errors = rep(NA,8)
test_errors = rep(NA,8)
Y_train=pro_1_train_data[train]
Y_test=pro_1_test_data$train
train_pred_matrix = model.matrix(train~., data = pro_1_train_data)

test_pred_matrix = model.matrix(train~., data = pro_1_test_data)
for (i in 1:8) {
    coefi = coef(best_sub, id = i)
    pred_train <- train_pred_matrix[,names(coefi)] %*% coefi
    train_errors[i] = mean((Y_train - pred_train)^2)
    pred_test <- test_pred_matrix[,names(coefi)] %*% coefi
    test_errors[i] = mean((Y_test - pred_test)^2)
}

test_errors

#0.2236606 0.2390604 0.2249300 0.2512954 0.2537141 0.2700974 0.2727801 0.2772596 
#first is the best

#10-fold
pro_1_train = sample(1:nrow(prostate), nrow(prostate)*0.90)
pro_1_test = -pro_1_train
pro_1_train_data = prostate[pro_1_train, ]
pro_1_test_data = prostate[pro_1_test, ]
best_sub = regsubsets(train~.,data = pro_1_train_data,method="exhaustive")
train_errors = rep(NA,8)
test_errors = rep(NA,8)
Y_train=pro_1_train_data[train]
Y_test=pro_1_test_data$train
train_pred_matrix = model.matrix(train~., data = pro_1_train_data)

test_pred_matrix = model.matrix(train~., data = pro_1_test_data)
for (i in 1:8) {
    coefi = coef(best_sub, id = i)
    pred_train <- train_pred_matrix[,names(coefi)] %*% coefi
    train_errors[i] = mean((Y_train - pred_train)^2)
    pred_test <- test_pred_matrix[,names(coefi)] %*% coefi
    test_errors[i] = mean((Y_test - pred_test)^2)
}

test_errors
#0.2086378 0.2217119 0.2263627 0.2200508 0.2280732 0.2175277 0.2139478 0.2097320
#first is the best


#boot
library(bootstrap)
library(boot)  #install.packages("bootstrap")
best_sub = regsubsets(train~.,data = prostate,method="exhaustive")
reg_summary=summary(best_sub)

beta.fit <- function(X,Y){
	lsfit(X,Y)	
}

beta.predict <- function(fit, X){
	cbind(1,X)%*%fit$coef
}

sq.error <- function(Y,Yhat){
	(Y-Yhat)^2
}
select = reg_summary$outmat
error_store <- c()
for (i in 1:8){
	# Pull out the model
	temp <- which(select[i,] == "*")
	
	res <- bootpred(prostate[,temp], prostate$train, nboot = 50, theta.fit = beta.fit, theta.predict = beta.predict, err.meas = sq.error) 
	error_store <- c(error_store, res[[3]])
	
}
#0.2136327 0.2135156 0.2106268 0.2179330 0.2147731 0.2188589 0.2301666 0.2334832
#third one is the best
#models using all
#         lcavol lweight age lbph svi lcp gleason pgg45 lpsa
#1  ( 1 ) " "    " "     "*" " "  " " " " " "     " "   " " 
#2  ( 1 ) " "    " "     " " " "  " " " " "*"     "*"   " " 
#3  ( 1 ) " "    " "     "*" " "  " " " " "*"     "*"   " " 
#4  ( 1 ) " "    " "     "*" " "  " " "*" "*"     "*"   " " 
#5  ( 1 ) " "    " "     "*" "*"  " " "*" "*"     "*"   " " 
#6  ( 1 ) " "    "*"     "*" "*"  " " "*" "*"     "*"   " " 
#7  ( 1 ) " "    "*"     "*" "*"  "*" "*" "*"     "*"   " " 
#8  ( 1 ) "*"    "*"     "*" "*"  "*" "*" "*"     "*"   " " 


#########2#################
library(tree)
library("rpart") #install.packages("rpart")
library(MASS)
set.seed(438)
setwd("G:/mou/UB/STA 545 Statistical Data Mining/Homework")
wine_data = read.csv('wine.data.txt',header = FALSE)

wine_train = sample(1:nrow(wine_data), nrow(wine_data)*0.80)
wine_test = -wine_train
wine_train_data = wine_data[wine_train, ]
wine_test_data = wine_data[wine_test, ]
model.control <- rpart.control(minsplit = 5, xval = 10, cp = 0)
wine_model <- rpart(V1~., data = wine_train_data, method = "class", control = model.control)
tree_pred = predict(wine_model, wine_test_data, type = "class")
mean(tree_pred != wine_test_data$V1)
#0.0278 pretty good fit

#to visulize 
plot(wine_model)
text(wine_model)

min_cp = which.min(wine_model$cptable[,4])
wine_model_pruned <- prune(wine_model, cp = wine_model$cptable[min_cp,1])
tree_pred = predict(wine_model_pruned, wine_test_data, type = "class")
mean(tree_pred != wine_test_data$V1)
#same error rates after prune
tree_pred = predict(wine_model_pruned, wine_train_data, type = "class")
mean(tree_pred != wine_train_data$V1)
#train error 0.0141
#similar results trying other parameters, yet when changing random number different results
#suggesting this model is superier than others (say using 5509) 

#using 
wine_model$frame
#to obtain node information 
table(wine_test_data$V1,tree_pred)



#4##########
spam = read.table("spam.data.txt")

spam_train = sample(1:nrow(spam), nrow(spam)*0.80)
spam_test = -spam_train
spam_train_data = spam[spam_train, ]
spam_test_data = spam[spam_test, ]
spam_train_data$V58 <- as.character(spam_train_data$V58)
spam_train_data$V58 <- as.factor(spam_train_data$V58)
ran_model = randomForest(V58~., data=spam_train_data,mtry=5,ntree=2500)
rab_model
#OOB 4.76%
ran_pred = predict(ran_model, spam_test_data, type = "class") 
 mean(ran_pred != spam_test_data$V58)
#0.0413

#Using 4 inputs:
ran_model = randomForest(V58~., data=spam_train_data,mtry=4,ntree=2500)
ran_model
#OOB 4.81%
ran_pred = predict(ran_model, spam_test_data, type = "class") 
mean(ran_pred != spam_test_data$V58)
#0.0402


#using 10:
ran_model = randomForest(V58~., data=spam_train_data,mtry=10,ntree=2500)
ran_model
#OOB 4.81%
ran_pred = predict(ran_model, spam_test_data, type = "class") 
mean(ran_pred != spam_test_data$V58)
#0.0423


#5######
library(neuralnet)
ls("package:neuralnet")
spam_train = sample(1:nrow(spam), nrow(spam)*0.80)
spam_test = -spam_train
spam_train_data = spam[spam_train, ]
spam_test_data = spam[spam_test, ]

n<-names(spam_train_data)
f <- as.formula(paste("V58 ~", paste(n[!n %in% "V58"], collapse = " + ")))
nn <- neuralnet(f, data=spam_train_data,hidden=5)
#
names(nn)
nn$result.matrix
round(nn$net.result[[1]])
nn_pred = round(compute(nn,spam_test_data[,1:57])$net.result[,1])
mean(spam_test_data$V58 != nn_pred)
#0.0738
#much more accurate than additive
