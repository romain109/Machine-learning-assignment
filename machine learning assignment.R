---
title: "machine learning assignment"
author: "Romain Molliet"
date: "25/05/2020"
output: html_document
---



## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:



  
## Import the Libraries
  

```{r}
 
library(caret)
library(rattle)
```


## Data Preparation


```{r}
rawdata <- read.csv("pml-training.csv", na.string = c("NA","#DIV/0!",""))
cleandata  <- rawdata[,colSums(is.na(rawdata))==0]
```





##Check if all the NAs are dropped and the left columns are ready for model training
```{r}
print(colSums(is.na(cleandata)))
```





##Remove the columns with unique values or have no predictive power
```{r}
data <- cleandata[,-c(1:7)]
```




##Get data prepared for training


```{r}
set.seed(12345)
inTrain <- createDataPartition(data$classe, p = 0.7, list = FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```




## Models Selection

## Rpart

##Set the cross validation and evaluate the rpart model 
```{r}
tc <- trainControl(method = "cv", number = 10)
mod_rpart <- train(classe ~ ., data = training, method = "rpart", trControl = tc)
fancyRpartPlot(mod_rpart$finalModel)

```



##The result shows that rpart method is not a good model since the purity in the leaves nodes are too low

## Clustering

##The classe have 5 unique values, so set k-centers to 5
```{r}
kmean <- kmeans(subset(training,select = -c(classe)), centers = 5)
training$cluster <- as.factor(kmean$cluster)
table(kmean$cluster, training$classe)
```



##The table shows clustering is not a good method as well since we cannot tell which group is belonged to which class.

```{r}
mod_cl <- train(cluster ~., data=training, method = "rpart")
pred_cl <- predict(mod_cl,testing)
table(pred_cl,testing$classe)
```



##The result derived from the prediction model further confirmed my opinion

## Delete the redundant column
```{r}
training <- training[,-ncol(training)]
```




## Since Random Forrest and Boosting and be good prediction model, try to minimise the number of features to lower the training time

```{r}
training.pca <- prcomp(training[,-ncol(training)], center = TRUE, scale. = TRUE)
summary(training.pca)
```



##The first 18 principal component analysis variable explain 90% of the variance.

## Convert the dataset to PCA dataset
```{r}
preProc <- preProcess(training[,-ncol(training)], method = "pca", pcaComp = 18)
training_pca <- predict(preProc, training[,-ncol(training)])
training_pca$classe <- training$classe
testing_pca <- predict(preProc, testing[,-ncol(testing)])
testing_pca$classe <- testing$classe
```




## Setup cross-validation 
```{r}
tc <- trainControl(method = "cv", number = 10)
```



## Train models
```{r}

mod_rf <- train(classe ~., data = training_pca, method = 'rf')
mod_gbm <- train(classe ~., data = training_pca, method = 'gbm',verbose  = FALSE)
mod_lda <- train(classe ~., data = training_pca, trControl = tc, method = 'lda')

```


## Get the prediction result
```{r}
pred_rf <- predict(mod_rf, testing_pca)
pred_gbm <- predict(mod_gbm, testing_pca)
```




## Random Forrest
```{r}
cv_rf <- confusionMatrix(pred_rf, as.factor(testing$classe))$overall[1]
print(cv_rf)
```




## Boosted Regression
```{r}
cv_gbm <- confusionMatrix(pred_gbm, as.factor(testing$classe))$overall[1]
print(cv_gbm)

```



## Linear discriminant analysis

cv_lda <- confusionMatrix(pred_lda, as.factor(testing$classe))$overall[1]
print(cv_lda)


## Combine the models with the weighting equal to each model's accurary


```{r}
predDF <- data.frame(pred_rf, pred_gbm, pred_lda, classe = testing$classe)
weighting <- c(cv_rf, cv_gbm, cv_lda)
weighting <- as.numeric(weighting)
combpred <- vector()
for (r in 1:nrow(predDF)) {
  A <- sum((predDF[r,-4]=="A") * weighting)
  B <- sum((predDF[r,-4]=="B") * weighting)
  C <- sum((predDF[r,-4]=="C") * weighting)
  D <- sum((predDF[r,-4]=="D") * weighting)
  E <- sum((predDF[r,-4]=="E") * weighting)
  combpred <- c(combpred,which.max(c(A,B,C,D,E)))
  combpred <- factor(combpred, levels = 1:5, labels = c("A", "B", "C", "D", "E"))
}
```
  
  



## Evaluate the result
```{r}
confusionMatrix(combpred, as.factor(testing$classe))
```




## The random forrest model performs the best

## Prediction result on the test dataset
```{r}
rawtest <- read.csv("pml-testing.csv", na.string = c("NA","#DIV/0!",""))
cleantest  <- rawtest[,colSums(is.na(rawtest))==0]
finaltest <- cleantest[,-c(1:7)]
finaltest_pca <- predict(preProc, finaltest[,-ncol(finaltest)])
finalPred <- predict(mod_rf, finaltest_pca)
print(finalPred)

```



