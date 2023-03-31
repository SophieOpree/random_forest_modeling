##### R Script Sophie Oprée: Predicting Cross-Buying Behavior #####

# Data exploration ####

# Read the dataset ---------------------------------

# clear workspace
rm(list = ls())

# set working directory
wd <- # set path
setwd(wd)

# load data
xsell <- get(load("xsell.RData"))

# load packages
library(psych)
library('fastDummies')
library(dplyr)
library(ggcorrplot)
library(ggplot2)
library("ROSE")
library(randomForest)
library(e1071)
library(caret)
library(pROC)
library(pdp)
library(vip)



# Explore the dataset ---------------------------------

# show data
View(xsell)
head(xsell)
tail(xsell)
names(xsell)
str(xsell) # overview on all variables

# determine size of data frame
dim(xsell)

# get summary
summary(xsell)

describe(xsell)

# explore sample characteristics
# age
table(xsell$age)
barplot(table(xsell$age), col = "lightblue", xlab = "age", ylab = "frequency", ylim = c(0, 4000))
title(main = "Distribution of age")

# gender
proportions <- table(xsell$gender)/length(xsell$gender)
percentages <- proportions*100
View(percentages)


# create histogramm for target variable: xsell
table(xsell$xsell)
prop.table(table(xsell$xsell))
ggplot(xsell)+geom_histogram(mapping = aes(x=xsell, fill= xsell),
                             stat = 'count', fill = "lightblue")
# distribution of xsell: 10% bought consumer loan



# Data preparation ---------------------------------

# check which variables contain missing variables
any(is.na(xsell))
colSums(is.na(xsell))
which(colSums(is.na(xsell))>0)
names(which(colSums(is.na(xsell))>0))

# replace missing values by the mean of the non-missing values
xsell$vol_eur_inflows[is.na(xsell$vol_eur_inflows)] <- round(mean(xsell$vol_eur_inflows, na.rm = TRUE))
xsell$vol_eur_outflows[is.na(xsell$vol_eur_outflows)] <- round(mean(xsell$vol_eur_outflows, na.rm = TRUE))
xsell$ext_city_size[is.na(xsell$ext_city_size)] <- round(mean(xsell$ext_city_size, na.rm = TRUE))
xsell$ext_house_size[is.na(xsell$ext_house_size)] <- round(mean(xsell$ext_house_size, na.rm = TRUE))
xsell$ext_purchase_power[is.na(xsell$ext_purchase_power)] <- round(mean(xsell$ext_purchase_power, na.rm = TRUE))

# check if all missing values have been replaced
any(is.na(xsell))
# checked


# add new "overdraft" variable: if the client has used the overdraft within 90 days
xsell$overdraft <- ifelse(xsell$nr_days_when_giro_below_0 >0, 1, 0)
table(xsell$overdraft)
aggregate(xsell ~ overdraft, data=xsell, FUN="mean") # check how important that variable is


# recode variables: explore levels and distribution
table(xsell$gender) # gender
levels(xsell$gender)
table(xsell$marital_status) # marital status
levels(xsell$marital_status)
table(xsell$occupation) # occupational status
levels(xsell$occupation)


# create dummy variables

# create marital status dummy
xsell$married_dummy <- ifelse(xsell$marital_status == "VH", 1, 0)
table(xsell$married_dummy)
# married = 1, not married = 0

# create occupation dummy
xsell$occupation_dummy <- ifelse(xsell$occupation == "Angestellter", 1, 0)
table(xsell$occupation_dummy)
# Angestellter = 1, not Angestellter = 0

# create gender dummy variables (not binary in this case)

xsell <- dummy_cols(xsell, select_columns = 'gender', remove_first_dummy = TRUE)

# first dummy was removed to avoid multicollinearity issues in models


# change long variable names

names(xsell)

xsell <- rename(xsell, tenure = customer_tenure_months)
xsell <- rename(xsell, vol_in = vol_eur_inflows)
xsell <- rename(xsell, vol_out = vol_eur_outflows)
xsell <- rename(xsell, login_d = logins_desktop)
xsell <- rename(xsell, login_m = logins_mobile)
xsell <- rename(xsell, married_s = marital_status)
xsell <- rename(xsell, recommender = member_get_member_recommender)
xsell <- rename(xsell, recommended = member_get_member_recommended)
xsell <- rename(xsell, debit = vol_eur_debit)
xsell <- rename(xsell, credit = vol_eur_credit)
xsell <- rename(xsell, products = nr_products)
xsell <- rename(xsell, relocations = nr_relocations)
xsell <- rename(xsell, giro_trx90 = nr_girocard_trx_90d)
xsell <- rename(xsell, visa_trx90 = nr_visacard_trx_90d)
xsell <- rename(xsell, days_giro_above_0 = nr_days_when_giro_above_0)
xsell <- rename(xsell, days_giro_below_0 = nr_days_when_giro_below_0)
xsell <- rename(xsell, city_size = ext_city_size)
xsell <- rename(xsell, house_size = ext_house_size)
xsell <- rename(xsell, purchase_pow = ext_purchase_power)
xsell <- rename(xsell, married_d = married_dummy) # d = dummy variable
xsell <- rename(xsell, occupation_d = occupation_dummy)
xsell <- rename(xsell, gender_F_d = gender_F)
xsell <- rename(xsell, gender_M_d = gender_M)
xsell <- rename(xsell, gender_MF_d = gender_MF)


# save data set with new variables
save(object = xsell, file = "xsell_clean_paper.Rdata")



# Data exploration ---------------------------------

# select only the numeric variables
xsell_numeric<-xsell[sapply(xsell, is.numeric)]

# create correlation matrix
correlations = round(cor(xsell_numeric), 2)
ggcorrplot(correlations,
           hc.order = TRUE,
           outline.color = "white",
           type = "lower", 
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"),
           show.legend = TRUE,
           title = "Correlation matrix")


# some bivariate analyses to explore relationships with target variable xsell

# age and xsell
xsell_agg <- aggregate(xsell ~ age, data=xsell, FUN="mean")
qplot(x=xsell_agg$age,y=xsell_agg$xsell,main="Cross-sell likelihood split by customer age",
      xlab="Age (years)", ylab="xsell", color=I("darkblue")) + theme_gray(base_size = 18)

# customer tenure and xsell
xsell_agg <- aggregate(xsell ~ tenure, data=xsell, FUN="mean")
qplot(x=xsell_agg$tenure,y=xsell_agg$xsell,main="Cross-sell likelihood split by customer tenure",
      xlab="Tenure (months)", ylab="xsell", ylim=c(0,0.2) , xlim=c(0,200), color=I("darkblue")) + theme_gray(base_size = 18)

# logins_desktop and xsell
xsell_agg <- aggregate(xsell ~ login_d, data=xsell, FUN="mean")
qplot(x=xsell_agg$login_d,y=xsell_agg$xsell, main="Cross-sell likelihood split by number of desktop logins",
      xlab="logins desktop", ylab="xsell", color=I("darkblue")) + theme_gray(base_size = 18)

# logins_mobile and xsell
xsell_agg <- aggregate(xsell ~ login_m, data=xsell, FUN="mean")
qplot(x=xsell_agg$login_m,y=xsell_agg$xsell, main="Cross-sell likelihood split by number of mobile logins",
      xlab="logins mobile", ylab="xsell", color=I("darkblue")) + theme_gray(base_size = 18)


# Check for character strings (relevant for further analyses)
xsell[sapply(xsell, is.character)]
# no character strings in dataset


# Recode categorical variables into a factor
# Create a function to identify binary variables
is.binary <- function(var){return(length(unique(var))==2)}

# Convert binary variables to factor
xsell[sapply(xsell,is.binary)] <- lapply(xsell[sapply(xsell, is.binary)], as.factor)
str(xsell)


# Create reduced dataset for further analyses (only containing variables used)
xsell_final <- xsell[, c(1, 2, 3, 4, 5, 8, 9, 24, 31, 32, 33, 34, 36, 30)]
names(xsell_final)


# Split dataset into training and validation -------
set.seed(87464) # fix random number generator seed for reproducibility
xsell_randomized <- xsell_final[order(runif(100000)),] #sort the data set randomly
xsell_valid <- xsell_randomized[1:20000, ]       # 20% / 2000 observations in the validation dataset
xsell_train <- xsell_randomized[20001:100000, ]   # 80% / 8000 in the training data set



# Random Forest: Model building -----

# Over-sample probability of xsell
# This is done to create a balanced training sample regarding the probability of xsell

set.seed(5427)
xsell_sampled <- ovun.sample(formula = xsell ~., data = xsell_train, method="both", p=0.5)

table(xsell_sampled$data$xsell)
xsell_balanced <- xsell_sampled$data
summary(xsell_balanced)


# Model Random Forest

# model specification (just for illustration)
model_01 <- xsell ~ income + age + acad_title + occupation_d + tenure + gender_M_d + calls + complaints + login_d + login_m + loan_mailing + overdraft + married_d


# compute random forest

set.seed(6574)

rf01 <- randomForest(xsell ~.,              # specified model
                     data = xsell_balanced, # dataset
                     importance = TRUE,     # calculate variable importance measure
                     replace = TRUE,        # sampling with replacement
                     keep.forest = TRUE)    # to use predict function afterwards
print(rf01)

# test error
err_test_rf01 <- mean(predict(rf01, xsell_valid) != xsell_valid$xsell)

# empirical error
err_emp_rf01 <- mean(predict(rf01, xsell_balanced) != xsell_balanced$xsell)



#### Model tuning ----

# plot development of ntree (number of trees in the forest)
set.seed(6574)
rf01_trace_ntree <- randomForest(xsell ~.,  # specified model
                                 data = xsell_balanced, # dataset
                                 replace=TRUE,          # sampling with replacement
                                 do.trace = 25)         # trace 
print(rf01_trace_ntree)
plot(rf01_trace_ntree)
# set ntree = 500 (equal to default value)


# optimize mtry (number of variables selected at each node)
set.seed(33323)
rf01_mtry_tune <- tuneRF(xsell_balanced[, -14], xsell_balanced[, 14], stepFactor=1.5, improve=0.05,
       trace=TRUE, plot=TRUE)
print(rf01_mtry_tune)
plot(rf01_mtry_tune)
# mtry = 6 as optimal value


# optimize nodesize (number of observations that a leaf of a tree must contain)

set.seed(12345)
nodesize.tuning <- c(2, 4, 6, 8, 10)

rf01_nodesize_tune <- tune.randomForest(xsell ~.,          # specified model
                               data = xsell_balanced,      # dataset
                               replace = TRUE,             # sampling with replacement
                               mtry = 6,                   # set optimal mtry
                               nodesize = nodesize.tuning, # optimize node size
                               ntree = 500)                # optimal number of trees
print(rf01_nodesize_tune)
# optimal value of nodesize = 2


# Use the tuning results to re-run the model

set.seed(33439)

rf02 <- randomForest(xsell ~.,              # specified model
                     data = xsell_balanced, # dataset
                     importance = TRUE,     # calculate variable importance measure
                     replace=TRUE,          # sampling with replacement
                     keep.forest = TRUE,    # to use predict function afterwards
                     mtry = 6,              # optimal number of variables at each split
                     ntree = 500,           # optimal number of trees
                     nodesize = 2)         # number of observations that a leaf must contain
                     
print(rf02)

# test error
err_test_rf02 <- mean(predict(rf02, xsell_valid) != xsell_valid$xsell)

# empirical error
err_emp_rf02 <- mean(predict(rf02, xsell_balanced) != xsell_balanced$xsell)



#### Compare models----

## Predictions

# predictions initial model
xsell_valid$pred_01 = predict(rf01, newdata = xsell_valid[-14], type = "class")
cm_01 = table(xsell_valid[,14], xsell_valid$pred_01)

# predictions tuned model
xsell_valid$pred_02 = predict(rf02, newdata = xsell_valid[-14], type = "class")
cm_02 = table(xsell_valid[,14], xsell_valid$pred_02)


## Confusion matrix

# confusion matrix for initial model
conf_rf01 <- confusionMatrix(xsell_valid$pred_01, xsell_valid$xsell)
print(conf_rf01)

#F1 score
conf_rf01$byClass["F1"] 


# confusion matrix for tuned model
conf_rf02 <- confusionMatrix(xsell_valid$pred_02, xsell_valid$xsell)
print(conf_rf02)

#F1 score
conf_rf02$byClass["F1"]


# compare models with ROC

# actual response
actual_response <-as.numeric(levels(xsell_valid$xsell))[xsell_valid$xsell]

# initial model
pred_rf_prob_01 <-predict(rf01, newdata=xsell_valid, type="prob")[,2] # initial model
roc_rf01 <-roc(actual_response, pred_rf_prob_01, percent = TRUE, plot = TRUE, print.auc = TRUE, grid = TRUE)
plot(roc_rf01)


ggroc(roc_rf01, colour = "darkblue")


# tuned model
pred_rf_prob_02 <-predict(rf02, newdata=xsell_valid, type="prob")[,2] # tuned model
roc_rf02 <-roc(actual_response, pred_rf_prob_02, percent = TRUE, plot = TRUE, print.auc = TRUE, grid = TRUE)
plot(roc_rf02)


#### Variable importance----


# method 1
importance(rf02)
varImpPlot(rf02, type = 1) # mean decrease accuracy
varImpPlot(rf02, type = 2) # mean decrease gini (slightly different results)
varUsed(rf02)


# method 2 with vip package

vip(rf02, bar = TRUE, horizontal = FALSE, include_type = TRUE,
    num_features = 13, aesthetics = list(color = "grey35", size = 0.8))


# partial plot

# marginal effect of income
partialPlot(rf02, xsell_balanced, x.var = "income", main = "Marginal effect: income")

# marginal effect of overdraft
partialPlot(rf02, xsell_balanced, x.var = "overdraft", main = "Marginal effect: overdraft")

# marginal effect of login_m
partialPlot(rf02, xsell_balanced, x.var = "login_m", main = "Marginal effect: mobile logins")

# marginal effect of login_d
partialPlot(rf02, xsell_balanced, x.var = "login_d", main = "Marginal effect: desktop logins")

# marginal effect of tenure
partialPlot(rf02, xsell_balanced, x.var = "tenure", main = "Marginal effect: tenure")

# marginal effect of age
partialPlot(rf02, xsell_balanced, x.var = "age", main = "Marginal effect: age")


# Shapley values

# separate independent variables in the model
X <- xsell_balanced[ , 1:13]

predictor <- Predictor$new(rf02, data=X, y=xsell_balanced$xsell)

# plot for different observations

# explain observation 1
shapley <- Shapley$new(predictor,x.interest=X[1, ])
shapley$results
plot(shapley)

# explain observation 2000
shapley <- Shapley$new(predictor,x.interest=X[2000, ])
shapley$results
plot(shapley)


