#a
library(ISLR)
dim(College)
#split 8:2
College.train <- College[1:as.integer(dim(College)[1]*0.8),]
College.test = College[as.integer(dim(College)[1]*0.8+1):as.integer(dim(College)[1]),]
#fit
fit <- lm(Apps ~  ., data=College.train)
#test error
prediction = predict(fit, College.test)
mean((College.test[, "Apps"] - prediction)^2)
#

#b
library(glmnet)
#remove APPs
drop<-c("Apps")
test <- College[,!(names(College) %in% drop)]
College.train_temp <-College.train[,!(names(College.train) %in% drop)]
College.test_temp <-College.test[,!(names(College.test) %in% drop)]

#Convert yes no
College.train_yn=College.train_temp
College.train_yn$Private<-ifelse(College.train_yn$Private=='Yes',1,0)


College.test_yn=College.test_temp
College.test_yn$Private<-ifelse(College.test_yn$Private=='Yes',1,0)
#fit
ridge_fit = cv.glmnet(as.matrix(College.train_yn), College.train[, "Apps"], alpha=0)
ridge_fit$lambda.min
#test
pred_ridge = predict(ridge_fit, newx=as.matrix(College.test_yn), s=ridge_fit$lambda.min)
mean((College.test[, "Apps"] - pred_ridge)^2)
#1058291


#d
lasso_fit = cv.glmnet(as.matrix(College.train_yn), College.train[, "Apps"], alpha=1)
pred_lasso = predict(lasso_fit, newx=as.matrix(College.test_yn), s=lasso_fit$lambda.min)
mean((College.test[, "Apps"] - pred_lasso)^2)
#1230480
predict(lasso_fit, newx=as.matrix(College.test_yn), s=lasso_fit$lambda.min,type="coefficients")
#all used by default setting


#e
library(pls)

#Convert yes no
College.train_pcr=College.train
College.train_pcr$Private<-ifelse(College.train_pcr$Private=='Yes',1,0)


College.test_pcr=College.test
College.test_pcr$Private<-ifelse(College.test_pcr$Private=='Yes',1,0)

pcr_fit = pcr(Apps~., data=College.train_pcr, scale=TRUE, validation="CV")
validationplot(pcr_fit, val.type="MSEP")
#finding k
train_error_store <- c()
test_error_store <- c()
for (i in 1:17) {
    pcr_pred_train = predict(pcr_fit, College.train_pcr, ncomp = i) 
    pcr_pred_test = predict(pcr_fit, College.test_pcr, ncomp = i) 
    pcr_train_error = mean((College.train[, "Apps"] - pcr_pred_train)^2)
    pcr_test_error = mean((College.test[, "Apps"] - pcr_pred_test)^2)
    train_error_store <- c(train_error_store, pcr_train_error)
    test_error_store <- c(test_error_store, pcr_test_error)
}
#smallest test error 1182249, k = 16


#f
pls_fit = plsr(Apps~., data=College.train_pcr, scale=TRUE, validation="CV")
validationplot(pls_fit, val.type="MSEP")
plstrain_error_store <- c()
plstest_error_store <- c()
for (i in 1:17) {
    pls_pred_train = predict(pls_fit, College.train_pcr, ncomp = i) 
    pls_pred_test = predict(pls_fit, College.test_pcr, ncomp = i) 
    pls_train_error = mean((College.train[, "Apps"] - pls_pred_train)^2)
    pls_test_error = mean((College.test[, "Apps"] - pls_pred_test)^2)
    plstrain_error_store <- c(plstrain_error_store, pls_train_error)
    plstest_error_store <- c(plstest_error_store, pls_test_error)
}
#smallest test error 1207047, k=9


#g
#blahblah
#do distribution plot for each
#hist(...)


########################
#2
########################


set.seed(5509)
train_data <-read.table('hw2_2_train.txt')
test_data <-read.table('hw2_2_test.txt')
train_train = sample(1:nrow(train_data), nrow(train_data)*0.80)
train_test = -train_train
#train data
train_train_data = train_data[train_train, ]
#test data
train_test_data = train_data[train_test, ]


lm_fit = lm(V86~., data = train_train_data)
lm_pred = predict(lm_fit, train_train_data)
#very high false negative rate, using logistic regression is much better in training (10) still 0 in testing
#a<-ifelse(lm_pred > 0.5,1,0) 
#
#mouge sb
#forward,backward fit
library(leaps)
regfit_fwd <- regsubsets(V86~., data=train_train_data, method = "forward",nvmax = 85)
regfit_bwd <- regsubsets(V86~., data=train_train_data, method = "backward",nvmax = 85)

#find best fit
test_matrix = model.matrix(V86 ~., data=train_test_data)
fwd_true_pos = rep(NA,84)
bwd_true_pos = rep(NA,84)
for (i in 1:84) {
    coef_fwd = coef(regfit_fwd, id = i)
    coef_bwd = coef(regfit_bwd, id = i)
    pred_fwd<-test_matrix[,names(coef_fwd)]%*%coef_fwd
    pred_bwd<-test_matrix[,names(coef_bwd)]%*%coef_bwd
    pred_fwd_bool <- ifelse(pred_fwd>0.5,1,0)
    pred_bwd_bool <- ifelse(pred_bwd>0.5,1,0)
    fwd_true_pos[i] = length(which(pred_fwd_bool == train_test_data$V86 & pred_fwd_bool == 1))
    bwd_true_pos[i] = length(which(pred_bwd_bool == train_test_data$V86 & pred_bwd_bool == 1))  
#    fwd_true_pos[i] = length(which(pred_fwd_bool == train_test_data$V86 ))
#    bwd_true_pos[i] = length(which(pred_bwd_bool == train_test_data$V86 ))
#    fwd_true_pos[i] = length(which(pred_fwd_bool == 1 ))
#    bwd_true_pos[i] = length(which(pred_bwd_bool == 1 ))
}
#all 0s

#ridge lasso
ridge_fit = cv.glmnet(as.matrix(train_train_data)[,1:85], train_train_data$V86, alpha=0)
lasso_fit = cv.glmnet(as.matrix(train_train_data)[,1:85], train_train_data$V86, alpha=1)
ridge_fit$lambda.min
#test
pred_ridge = predict(ridge_fit, newx=as.matrix(train_test_data)[,1:85], s=ridge_fit$lambda.min)
pred_lasso = predict(lasso_fit, newx=as.matrix(train_test_data)[,1:85], s=lasso_fit$lambda.min)

pred_ridge_bool <- ifelse(pred_ridge>0.5,1,0)
pred_lasso_bool <- ifelse(pred_lasso>0.5,1,0)
length(which(pred_ridge_bool==train_test_data$V86 & pred_ridge_bool ==1))
length(which(pred_lasso_bool==train_test_data$V86 & pred_lasso_bool ==1))

#both test 0





#######################
#3
#######################


#a
#generate random x beta 
for_x=sample(1:100,20*1000,replace = TRUE)
X<- matrix(for_x,ncol = 20)
beta=sample(1:10, 20,replace=TRUE)
beta[1:5] = 0
epsilon = 666
Y = X %*% beta+epsilon
#

sam3 = sample(seq(1000), 100, replace = FALSE)
Y_train = y[sam3, ]
Y_test = y[-sam3, ]
X_train = x[sam3, ]
X_test = x[-same, ]

train_data = data.frame(X_train, Y_train)

test_data = data.frame(X_test, Y_test)
best_sub = regsubsets(Y_train~., data=train_data, nvmax = 20, method = "exhaustive")

train_pred_matrix = model.matrix(Y_train ~., data = train_data)
test_pred_matrix = model.matrix(Y_test ~., data = test_data)

train_errors = rep(NA,20)
test_errors = rep(NA,20)




for (i in 1:20) {
    coefi = coef(best_sub, id = i)
    pred_train <- train_pred_matrix[,names(coefi)] %*% coefi
    train_errors[i] = mean((Y_train - pred_train)^2)
    pred_test <- test_pred_matrix[,names(coefi)] %*% coefi
    test_errors[i] = mean((Y_test - pred_test)^2)
}

#for training MES, using all is minimized For testing, using 15 is minimal, exactly the number of feature used
