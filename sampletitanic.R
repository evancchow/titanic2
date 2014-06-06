# R script to analyze the titanic data.
# To execute multiple commands: > cmd1; cmd2; cmd3 ...
# Easy to keep running script @ RStudio (w/up arrow):
# > source("titanic.R"); head(train)
# NOTE: for predict() test and train set must have same var names in dataframe

# Load the Titanic data. Rm less important columns, drop rows with NA, and
# binarize the Sex column.
drops <- c("Name", "Ticket", "Cabin", "PassengerId")
data <- read.csv("train.csv")
data <- data[,!names(data) %in% drops]
data <- data[complete.cases(data),]
data$Sex <- ifelse(data$Sex=="male",1,0)
data$Embarked <- NULL # rm "Embarked" column for knn

# partition into train/test sets
train <- data[1:500,]
test <- data[501:length(data),]

trainx <- train[,c(2:length(train))]
testx <- test[,c(2:length(test)])
trainy <- train[,c(1)]
testy <- test[,c(1)]

# Stop condition if only want to read data.
stop("No error: finished reading and partitioning data.")

#----- Logistic Regression -----#

# Training error rate
logistic <- glm(Survived~Sex+Age+Pclass,data=train,family=binomial)
logitprobs <- predict(logistic, type="response")
logitpreds <- rep(0, length(logitprobs))
logitpreds[logitprobs > 0.5] <- 1
table(logitpreds, train$Survived)
mean(logitpreds==train$Survived)

# Test error rate
testprobs <- predict(logistic, test, type="response")
testpreds <- rep(0, length(testprobs))
testpreds[testprobs > 0.5] <- 1
table(testpreds, test$Survived)
mean(testpreds==test$Survived)

#----- K-Nearest Neighbors -----#

### Not entirely sure how to use predict() for knn
# library(class)
# knn <- knn(trainx, testx, trainy, k=3, prob=TRUE)
# knn.probs <- attributes(knn)$prob
# knn.pred <- rep(0, length(knn.probs))
# knn.pred[knn.probs > 0.5] <- 1
# table(knn.pred)

