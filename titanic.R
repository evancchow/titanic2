# R script to analyze the titanic data.
# To execute multiple commands: > cmd1; cmd2; cmd3 ...
# Easy to keep running script @ RStudio (w/up arrow):
# > source("titanic.R"); head(train)
# NOTE: for predict() test and train set must have same var names in dataframe
# (stop script with "stop(text here)")

# Load the Titanic data. Rm less important columns, drop rows with NA, and
# binarize the Sex column. I.e. preprocessing.
drops <- c("Name", "Ticket", "Cabin", "PassengerId")
data <- read.csv("train.csv")
data <- data[,!names(data) %in% drops]
data <- data[complete.cases(data),]
data$Embarked <- NULL # rm "Embarked" column for knn

# Partition original training data into train and held-out validation set
# randomly chooses training rows with 3/4 prob of choosing (set to TRUE)
train.ixs <- sample(c(TRUE,TRUE,TRUE,FALSE), nrow(data), rep=TRUE)
test.ixs <- (!train.ixs)
training.data <- data[train.ixs,]
testing.data <- data[test.ixs,]
train.x <- training.data[,!names(training.data) %in% c("Survived")]
train.y <- training.data$Survived
test.x <- testing.data[,!names(testing.data) %in% c("Survived")]
test.y <- testing.data$Survived

# Using best subset selection, plotting shows R^2 increases when you go from 
# 1 to 2 vars (and also when go from 2 to 3) but converges after that. Also
# summary shows that three most important variables (most important first)
# are Sex, then Pclass, then Age. (at least on the training data)
set.seed(1)
library(leaps)
reg.fit <- regsubsets(Survived~., data=training.data)
reg.summary <- summary(reg.fit)
plot(reg.summary$rsq) # starts converging after n vars = 3
summary(reg.fit) # best three vars (most important 1st): Sex, Pclass, Age

# Fit logistic regression model to best 3 vars of training data. Remember
# that logistic regression yields probabilities of survival (rather than
# just simple yes/no), so youn need to create a vector that has 1's for 
# P(Survival) > 0.5 or some threshold, and 0's elsewhere. Also, remember for
# the future that you can do glm(..., subset=train).
logistic <- glm(Survived~Sex+Pclass+Age, data=training.data, family="binomial")

### How did you do on the training set? ###
train.probs <- predict(logistic, train.x, type="response")
train.pred <- rep(0, nrow(train.x))
train.pred[train.probs > 0.5] <- 1
mean(train.pred==train.y)

# See how did on validation set
test.probs <- predict(logistic, test.x, type="response")
test.preds <- rep(0, nrow(test.x))
test.preds[test.probs > 0.5] <- 1
table(test.preds==test.y)
mean(test.preds==test.y)
