data <- read.csv("./data/training.csv")
valid <- read.csv("./data/Valid.csv")


inTrain <- createDataPartition(y=data$classe,
                               p=0.7, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
dim(training); dim(testing)
columns <- c(8:11, 37:49,60:68,84:86, 102, 113:124,140,151:160)
training <- training[,columns]
testing <- testing[,columns]

set.seed(123)

modFitRf <- train(classe~.,data=training, method="rf")
prediction_rf <- predict(modFitRf,newdata=testing)
confusionMatrix(prediction_rf,testing$classe)