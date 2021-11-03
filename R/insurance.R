library(ggplot2) #plot
library(ggpubr) #layout dei plot
library(corrplot) #correlationmatrix
library(splitstackshape) #suddivisione test/train stratificata
library(rpart) #alberi di decisione
library(caret)
library(unbalanced) #bilanciamento
library(pROC) #ROC
library(e1071) #naive bayes
library(xgboost) #boosting
library(Matrix) #manipolazione matrice

insurance <- read.csv("C:/Users/placi/PycharmProjects/insurance/insurance.csv")
summary(insurance)

#PRE-PROCESSING
#gender
insurance[which(insurance$Gender == "Male"),]$Gender = 1
insurance[which(insurance$Gender == "Female"),]$Gender = 0
colnames(insurance)[2] = "Male"
insurance$Male <- as.factor(insurance$Male)
#vehicle_age
insurance[which(insurance$Vehicle_Age == "< 1 Year"),]$Vehicle_Age <- -1
insurance[which(insurance$Vehicle_Age == "1-2 Year"),]$Vehicle_Age <- 0
insurance[which(insurance$Vehicle_Age == "> 2 Years"),]$Vehicle_Age <- 1
insurance$Vehicle_Age <- as.numeric(insurance$Vehicle_Age)
#vehicle_damage
insurance[which(insurance$Vehicle_Damage == "Yes"),]$Vehicle_Damage <- 1
insurance[which(insurance$Vehicle_Damage == "No"),]$Vehicle_Damage <- 0
insurance$Vehicle_Damage <- as.factor(insurance$Vehicle_Damage)
#categoriche e binarie -> factor
insurance$Driving_License <- as.factor(insurance$Driving_License)
insurance$Region_Code <- as.factor(insurance$Region_Code)
insurance$Previously_Insured <- as.factor(insurance$Previously_Insured)
insurance$Policy_Sales_Channel <- as.factor(insurance$Policy_Sales_Channel)
insurance$Response <- as.factor(insurance$Response)

#DISTRIBUZIONI DELLE FREQUENZE
#age
age_plot <- ggplot(insurance, aes(x=Age, fill=Response))+
  geom_histogram(alpha=0.4,position="identity")+
  xlab("Età")+
  scale_y_continuous("Numero di osservazioni") +
  guides(fill=guide_legend(title=NULL)) + 
  scale_fill_discrete(labels=c("Negativi","Positivi")) +
  theme(legend.position = c(1,1),legend.justification=c(1,1))
#vintage
vintage_plot <- ggplot(insurance, aes( x=Vintage, fill=Response))+
  geom_histogram(alpha=0.4,position="identity")+
  xlab("Vintage")+
  scale_y_continuous("Numero di osservazioni")
#annual_premium
premium_plot <- ggplot(insurance, aes( x=Annual_Premium, fill=Response))+
  geom_histogram(alpha=0.4,position="identity")+
  xlab("Premio Annuale")+
  scale_y_continuous("Numero di osservazioni") +
  scale_x_continuous(limits = c(0, 110000))

ggarrange(age_plot, ggarrange(vintage_plot, premium_plot, ncol = 2, legend = "none"),
          nrow = 2, common.legend = T, legend = "bottom")

#MATRICE DI CORRELAZIONE
correlationMatrix <- stats::cor(insurance[c(3, 7, 9, 11)])
corrplot(correlationMatrix, method="circle", type="lower", tl.col="black")

#DIMOSTRAZIONE DELLO SBILANCIAMENTO
freq <- as.data.frame(table(insurance$Response))
colnames(freq)[1] = "Response"
freq$perc <- prop.table(freq$Freq)
print(freq)
ggplot(freq, aes(x = "", y = perc, fill = Response)) +
  geom_col(color = "black") +
  coord_polar(theta = "y") +
  xlab("") + 
  ylab("") +
  ggtitle("Frequenza delle classi") +
  theme(plot.title = element_text(hjust = 0.5))

#BOXPLOT
par(mfrow=c(1,3))
#age
boxplot(insurance$Age, main="Age")
#annual_premium
boxplot(insurance$Annual_Premium, main="Annual_Premium")
#vintage
boxplot(insurance$Vintage, main="Vintage")
par(mfrow=c(1,1))

#SEPARAZIONE STRATIFICATA TEST/TRAIN
set.seed(1)
train <- as.data.frame(stratified(insurance, c('Response'), 0.7))
test <- insurance[which(!(insurance$id %in% train$id)),]
insurance <- insurance[, -1]
train <- train[, -1]
test <- test[,-1]

#ALBERO DI DECISIONE SU DATASET DI TRAINING SBILANCIATO
set.seed(1)
tree <- rpart(Response~., train, method = "class", control = rpart.control(cp = 0.00001))
prediction_unbalanced <- predict(tree, test, type = "class")
confusionMatrix(as.factor(prediction_unbalanced),as.factor(test$Response))

#RIBILANCIAMENTO
#random undersampling
set.seed(1)
rem <- ubUnder(train[,1:11], train$Response, perc=14, method = "percUnder")
casual.balance.train <- data.frame(rem$X, rem$Y)
casual.balance.train$rem.Y <- NULL
names(casual.balance.train)[11] <- "Response"
table(casual.balance.train$Response)
#SMOTE + random undersampling
set.seed(1)
SMOTE <- ubBalance(train[,-c(11)], train[,11],type="ubSMOTE", positive = 1, 
                   percOver = 100, percUnder = 280, verbose = T)
SMOTE_train <- SMOTE$X
SMOTE_train$Response <- SMOTE$Y
table(SMOTE_train$Response)

#ALBERI DI DECISIONE
#undersampling
set.seed(1)
tree <- rpart(Response~., casual.balance.train, method = "class",
              control = rpart.control(cp = 0.00001))
prediction_casual_balance <- predict(tree, test, type = "class")
confusionMatrix(as.factor(prediction_casual_balance),as.factor(test$Response))
#SMOTE + undersampling
set.seed(1)
tree <- rpart(Response~., SMOTE_train, method = "class",
              control = rpart.control(cp = 0.0001))
prediction_smote <- predict(tree, test, type = "class")
confusionMatrix(as.factor(prediction_smote),as.factor(test$Response))
#ROC
par(pty = "s")
plot.roc(as.numeric(test$Response), as.numeric(prediction_unbalanced), col="red")
lines.roc(as.numeric(test$Response), as.numeric(prediction_casual_balance), col="blue")
lines.roc(test$Response, as.numeric(prediction_smote), col="green")

#NAIVE-BAYES
#sbilanciato
bayes_bil <- naiveBayes(Response ~ ., data = casual.balance.train)
b_prediction_unbalanced <- predict(bayes_bil, newdata = test)
confusionMatrix(as.factor(b_prediction_unbalanced),as.factor(test$Response))
legend("bottomright", legend = c("Unbalanced", "Random Undersampling", "SMOTE"),
       lty=1, col=c("red", "blue", "green"), cex = 0.5)
#undersampling
bayes <- naiveBayes(Response ~ ., data = train)
b_prediction_casual_balance <- predict(bayes, newdata = test)
confusionMatrix(as.factor(b_prediction_casual_balance),as.factor(test$Response))
#SMOTE + undersampling
bayes_bil <- naiveBayes(Response ~ ., data = SMOTE_train)
b_prediction_smote <- predict(bayes_bil, newdata = test)
confusionMatrix(as.factor(b_prediction_smote),as.factor(test$Response))
#ROC
plot.roc(as.numeric(test$Response), as.numeric(b_prediction_unbalanced), col="red")
lines.roc(as.numeric(test$Response), as.numeric(b_prediction_casual_balance), col="blue")
lines.roc(as.numeric(test$Response), as.numeric(b_prediction_smote), col="green")
legend("bottomright", legend = c("Unbalanced", "Random Undersampling", "SMOTE"),
       lty=1, col=c("red", "blue", "green"), cex = 0.5)

#SVM
weights <- c(1,6)
names(weights) <- c(0, 1)
set.seed(1)
support <- svm(formula = Response ~ ., data = train[, 1:11], kernel = "linear", class.weights = weights, scale = TRUE)
svm_prediction <- predict(support, newdata = test)
confusionMatrix(as.factor(svm_prediction),as.factor(test$Response))
plot.roc(as.numeric(test$Response), as.numeric(svm_prediction), col="red")

#XGBOOST
#preparazione dataset
xgb_preprocess <- function(df)
{
  df.xgb <- df[,c(1,2,3,5,6,7,8,10,11)]
  df.xgb$Male <-  as.numeric(as.character(df.xgb$Male))
  df.xgb$Driving_License <- as.numeric(as.character(df.xgb$Driving_License))
  df.xgb$Vehicle_Damage <- as.numeric(as.character(df.xgb$Vehicle_Damage))
  df.xgb$Previously_Insured <- as.numeric(as.character(df.xgb$Previously_Insured))
  df.xgb$Response <- as.numeric(as.character(df.xgb$Response))
  df.xgb$Age <- as.numeric(df.xgb$Age)
  df.xgb$Vintage <- as.numeric(df.xgb$Vintage)
  df.xgb <- as.matrix(df.xgb)
  return(df.xgb)
}
train.xgb = xgb_preprocess(train)
test.xgb = xgb_preprocess(test)
test.xgb <- test.xgb[,-9]
#generazione del modello
weight <- ifelse(train$Response==1, 6,1)
set.seed(1)
bst <- xgboost(data = train.xgb[,-9], label = train.xgb[,9], 
               nthread = -1, nrounds = 100, weight = weight, verbose=0)
xgb_prediction <- predict(bst, test.xgb)
xgb_prediction <- as.numeric(xgb_prediction > 0.5)
confusionMatrix(as.factor(xgb_prediction),as.factor(test$Response))
par(pty = "s")
#ROC
plot.roc(as.numeric(test$Response), as.numeric(xgb_prediction), col="red")