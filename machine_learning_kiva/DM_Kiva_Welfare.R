library(dplyr)
library(car)
library(psych)
library(rpart)
library(SnowballC)
library(dplyr)
library(e1071)
library(base)
library(Matrix)
library(data.table)
library(stats)
library(caret)
library(MASS)
library(ggplot2)
library(psych)
library(partykit)
library(lattice)
library(klaR)
library(C50)
library(gmodels)
library(Matrix)
library(class)
library(NLP)
library(tm)
library(wordcloud)
library(vcd)
library(adabag)
library(ipred)
library(neuralnet)
library(class)


###################### Unsupervised Learning on Welfare: K-Means #############################

#------------------ Dimensionality Reduction: Integrated Kiva Loans Dataset
###get factor and string cols and make those empty ones NA
kloans <- read.csv(file.choose(), header = TRUE, na.strings = c("", " ", "NA"))
#it was the kiva_loans_mpi_theme_indicators_stats file

#remove csv rowcount
kloans <- kloans[,-1]
#remove use(6), dates(11,12,13,18), tags(16), country.y(24), partnerid(30), region(20), country_name(22)
kloans <- kloans[,-c(6,11,12,13,18,16,24,29,20,21,22)]
kloans <- kloans[,-17]
names(kloans)[7] <- "country"
#country code(6) and country code3(14) --> remove
kloans$partner_id <- ifelse(is.na(kloans$partner_id), "None", kloans$partner_id)
kloans <- kloans[,-c(6,14)]
names(kloans)

#remove all rows that have at least one NA in a column
k <- na.omit(kloans)
dim(k)
#[1] 575140     33

#make sure all are the type they should be
k$partner_id <- as.factor(k$partner_id)
k$id <- as.factor(k$id)

#reduce to only get numerical values
# numerical: 2,3,10,11,16,18:27,29:35
# categorical: 4,5,6,7,8,9,12,13,14,15,17,27
kstats2 <- k[,c(2,3,9,10,16:25,27:33)]
kcorrelation <- cor(kstats2)

#scale everything
kstats_z2 <- as.data.frame(lapply(kstats2, scale))

#PCA & Screeplot to see what we should exclude
kstats_pr2 <- prcomp(kstats_z2)
kstats_pr2
summary(kstats_pr2)

#Factor Analysis to see if there are any factors we can exclude
kfa <- fa(r = kcorrelation, nfactors = 2, rotate = "oblimin", fm="ml")
kfa

### Summary: Funded Amount, Lender Count, Term in Months explain very little variation
###          Country Indicators and Country Stats are inversely correlated



#------------------ K-Means --> Choosing right K

set.seed(123)
wss <- sapply(1:k.max, function(k){kmeans(kstats_z2,k)$tot.withinss})
plot(1:k.max, wss, type = "b", pch=19, frame=FALSE,xlab = "Number of Clusters",ylab = "Total w/in Clusters Sum of Squared")
plot(1:k.max, wss, type = "b", pch=19, frame=FALSE,xlab = "Number of Clusters",ylab = "Total w/in Clusters Sum of Squared");abline(v=6)
plot(1:k.max, wss, type = "b", pch=19, frame=FALSE,xlab = "Number of Clusters",ylab = "Total w/in Clusters Sum of Squared");abline(v=5)

kstats_z2_red <- kstats_z2[,-c(1:4)]
names(kstats_z2_red)
wss <- sapply(1:k.max, function(k){kmeans(kstats_z2_red,k)$tot.withinss})
plot(1:k.max, wss, type = "b", pch=19, frame=FALSE,xlab = "Number of Clusters",ylab = "Total w/in Clusters Sum of Squared")

kstats_z2_red_cluster <- kmeans(kstats_z2_red,4)
fviz_cluster(kstats_z2_red_cluster, data = kstats_z2_red, geom = "point", stand = TRUE, ellipse.type = "norm")
kstats_z2_red_cluster


### K = 5
kstats_z2_red_cluster5 <- kmeans(kstats_z2_red,5)
fviz_cluster(kstats_z2_red_cluster5, data = kstats_z2_red, geom = "point", stand = TRUE, ellipse.type = "norm")
kstats_z2_red_cluster5

### Choosing K = 5


#------------------ K-Means --> Adding Cluster Label to Dataset

k$cluster <- kstats_z2_red_cluster5$cluster

# change labels to more descriptive name
# 1 to 5 from most developed to least developed
k$cluster <- ifelse(k$cluster == 1, 2,
                    ifelse(k$cluster == 2, 4,
                           ifelse(k$cluster == 3, 5,
                                  ifelse(k$cluster== 4, 3,
                                         ifelse(k$cluster == 5, 1, k$cluster)))))
k$cluster <- as.factor(k$cluster)
aggregate(data = k , k$loan_amount~k$cluster, mean)
table(k$sector, k$cluster)


#------------------ K-Means --> Creating Training and Testing Datasets

k <- k[-1]
set.seed(123)
pd <- sample(2,nrow(k),replace=TRUE,prob=c(0.75,0.25))
ktrain <- k[pd == 1,]
ktest <- k[pd == 2,]

###################### Predicting Welfare: Naive Bayes #############################
# Reduce Dataset (w/o stats and indicators to see how accurate it is without those numbers)
ktrain_red <- ktrain[,-c(15:24,26:32)]
ktest_red <- ktest[,-c(15:24,26:32)]

# Create Classifier
ktrain_red_nb <- naiveBayes(ktrain_red$cluster~., data = ktrain_red)

# Predict
nb_pred <- predict(ktrain_red_nb, ktest_red[-16], type = "class")

# Evaluate
CrossTable(nb_pred, ktest_red$cluster, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted','actual'))
# Accuracy
31463+62191+9749+30904+7582
#[1] 141889
141889/143503
#[1] 0.9887528

confusionMatrix(nb_pred, ktest_red$cluster)
# Accuracy : 0.9888          
# 95% CI : (0.9882, 0.9893)
# No Information Rate : 0.4431          
# P-Value [Acc > NIR] : < 2.2e-16       
# Kappa : 0.984           
# Mcnemar's Test P-Value : NA


#------------------ Summary

# Accuracy: high; Kappa: High; Error Rate: Low
# Very accurate --> it can predict the cluster and therefore welfare of a loan requestor
# Question --> did all variables have the same impact on the apriori?
# Decision 1 --> do Decision Tree to see if there is a variable that has prevalence over others in classifying


###################### Predicting Welfare: Decision Tree #############################

#------------------ Including Country for first Decision Tree
ktrain_dt <- ktrain_red[,c(4,5,12,13,15,16)]
ktest_dt <- ktest_red[,c(4,5,12,13,15,16)]

# Create Classifier
k_dt <- C5.0(ktrain_dt[-6], ktrain_dt$cluster, trials = 3)
k_dt
summary(k_dt)

# Predict
dt_pred <- predict(k_dt, ktest_dt[-6])

# Evaluate
CrossTable(dt_pred, ktest_red$cluster, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted','actual'))
confusionMatrix(dt_pred, ktest_dt$cluster)
# Accuracy : 1        
# 95% CI : (1, 1)   
# No Information Rate : 0.4431   
# P-Value [Acc > NIR] : < 2.2e-16
# 
# Kappa : 1        
# Mcnemar's Test P-Value : NA

#------------------ Summary

# 100% accurate, BUT does not provide much information other than grouping all the countries
# that belong to a cluster. Did not take into consideration other variables that I would
# have liked to see how loan amount, sector, activity, and world region can help determine
# the welfare level where a loan is being requested. Country is an obivous predictor, good
# insight to have, but I want to explore more relationships of the dataset.

# Decision: Exclude country and country MPI to see if we can understand the role loan amount
# sector, activity, world region in predicting welfare.

#------------------ Excluding Country & Country MPI for Second Decision Tree
# Predict Welfare Based on World Region, Loan Amount, Activity, and Sector the Loan is being requested for
# Create Classifier
set.seed(123)
kwf_dt2 <- C5.0(ktrain_red$cluster~loan_amount+activity+sector+world_region, data = ktrain_red)
kwf_dt2
summary(kwf_dt2)

# Predict
kwf_dt2pred <- predict(kwf_dt2, ktest_red[,-c(16,1,5,6,7,8,9,10,11,13,14,15)])

#Evaluate
CrossTable(kwf_dt2pred, ktest_red$cluster, prop.r = FALSE, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted','actual'))
confusionMatrix(kwf_dt2pred, ktest_red$cluster)
#                Accuracy : 0.907           
#                  95% CI : (0.9055, 0.9085)
#     No Information Rate : 0.4431          
#     P-Value [Acc > NIR] : < 2.2e-16       
#                                           
#                   Kappa : 0.8677          
#  Mcnemar's Test P-Value : NA


#------------------ Summary

# Accuracy: High; Kappa: High (still above 85%); Error Rate: low (below 10%)
# Very Accurate --> performs well in classifying welfare based on world region, activity,
# sector, and loan amount

# Decision: Try to improve accuracy: Try Bagging and Boosting for Decision Trees


############################# Cross Validation on Decision Tree ##########################
# Combine Train and Test & remove any unused factor levels (it gives error on unused factor levels)
t <- k
t <- t %>% mutate_if(is.factor, as.character)
t$activity <- as.factor(t$activity)
t$sector <- as.factor(t$sector)
t$country <- as.factor(t$country)
t$partner_id <- as.factor(t$partner_id)
t$repayment_interval <- as.factor(t$repayment_interval)
t$borrower_gender <- as.factor(t$borrower_gender)
t$world_region <- as.factor(t$world_region)
t$cluster <- as.factor(t$cluster)
t_red <-t[,-c(15:24,26:32)]

# Number of Trials = 1
treectrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE, selectionFunction = "oneSE")
grid <- expand.grid(.model = "tree", .trials=c(1), .winnow="FALSE")
set.seed(123)
kwf_dt2_cv <- train(cluster~loan_amount+activity+sector+world_region, data = t_red, method="C5.0", trControl = treectrl, tuneGrid=grid, metric="Kappa")
kwf_bag_pred <- predict(kwf_dt2_cv, ktest_red[,-c(16,1,5,6,7,8,9,10,11,13,14,15)])
confusionMatrix(kwf_bag_pred, ktest_red$cluster)
# Accuracy : 0.9086          
# 95% CI : (0.9071, 0.9101)
# No Information Rate : 0.4431          
# P-Value [Acc > NIR] : < 2.2e-16       
# Kappa : 0.87 

#------------------ Summary
# Results in same accuracy as running the C50 once


# Number of Trials = 3
grid2 <- expand.grid(.model = "tree", .trials=c(3), .winnow="FALSE")
kwf_dt2_cv2 <- train(cluster~loan_amount+activity+sector+world_region, data = t_red, method="C5.0", trControl = treectrl, tuneGrid=grid2, metric="Kappa")
kwf_dt2_cv2
kwf_bag_pred2 <- predict(kwf_dt2_cv2, ktest_red[,-c(16,1,5,6,7,8,9,10,11,13,14,15)])
confusionMatrix(kwf_bag_pred2, ktest_red$cluster)
# Accuracy : 0.9054          
# 95% CI : (0.9039, 0.9069)
# No Information Rate : 0.4431          
# P-Value [Acc > NIR] : < 2.2e-16       
# Kappa : 0.8654          
# Mcnemar's Test P-Value : NA 

#------------------ Summary
# Results in slightly lower accuracy than running only 1 trial

############################# Bagging for Decision Tree ###################################
set.seed(123)
mybag <- bagging(cluster~loan_amount+activity+sector+world_region, data = ktrain_red, nbagg = 10)
bag_pred <- predict(mybag, ktest_red[,-c(16,1,5,6,7,8,9,10,11,13,14,15)])
table(bag_pred, ktest_red$cluster)

############################# Bagging for Decision Tree w/ CV ############################
treectrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
set.seed(123)
kwf_dt2_bag <- train(cluster~loan_amount+activity+sector+world_region, data = ktrain_red, method="treebag", trControl = treectrl)
kwf_bag_pred <- predict(kwf_dt2_bag, ktest_red[,-c(16,1,5,6,7,8,9,10,11,13,14,15)])









