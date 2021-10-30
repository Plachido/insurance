library(rpart) #per alberi di decisione
library(rattle)#per plot di alberi
library(corrplot) #per correlationmatrix
library(parallel) #per parallelizzare
library(caret) #per confusionmatrix
library(unbalanced) #per bilanciamento
library(pROC) #per curva ROC
library(splitstackshape) #per suddivisione test/train stratificata
library(e1071) #per bayes
library(adabag) #per boosting
library(gridExtra) #per layout
library(mlbench) #per plot
library(ggplot2)
library(ggpubr)
library(RColorBrewer)
insurance <- read.csv("D:/LUMSA/Artificial intelligence/PROGETTO/insurance.csv")
#D:/LUMSA/Artificial intelligence/PROGETTO/insurance.csv
##############Manipolazione dei dati
#Operazioni sui dati
insurance[which(insurance$Gender == "Male"),]$Gender = 1
insurance[which(insurance$Gender == "Female"),]$Gender = 1
colnames(insurance)[2] = "Male"
insurance$Male <- as.factor(insurance$Male)
insurance[which(insurance$Vehicle_Age == "< 1 Year"),]$Vehicle_Age <- -1
insurance[which(insurance$Vehicle_Age == "1-2 Year"),]$Vehicle_Age <- 0
insurance[which(insurance$Vehicle_Age == "> 2 Years"),]$Vehicle_Age <- 1
insurance$Vehicle_Age <- as.numeric(insurance$Vehicle_Age)
insurance[which(insurance$Vehicle_Damage == "Yes"),]$Vehicle_Damage <- 1
insurance[which(insurance$Vehicle_Damage == "No"),]$Vehicle_Damage <- 0
insurance$Vehicle_Damage <- as.factor(insurance$Vehicle_Damage)
insurance$Driving_License <- as.factor(insurance$Driving_License)
insurance$Region_Code <- as.factor(insurance$Region_Code)
insurance$Previously_Insured <- as.factor(insurance$Previously_Insured)
insurance$Policy_Sales_Channel <- as.factor(insurance$Policy_Sales_Channel)
insurance$Response <- as.factor(insurance$Response)

#Distribuzioni delle frequenze
#Frequenze Age
age_plot <- ggplot(insurance, 
               aes(x=Age, fill=Response))+
    geom_histogram(alpha=0.4,position="identity")+
    xlab("Età")+
    scale_y_continuous("Numero di osservazioni") + guides(fill=guide_legend(title=NULL)) + 
  scale_fill_discrete(labels=c("Negativi","Positivi")) +
  theme(legend.position = c(1,1),legend.justification=c(1,1))

#Frequenze Vintage
vintage_plot <- ggplot(insurance, 
             aes( x=Vintage, fill=Response))+
  geom_histogram(alpha=0.4,position="identity")+
  xlab("Vintage")+
  scale_y_continuous("Numero di osservazioni")

#Frequenze annual_premium
premium_plot <- ggplot( insurance, 
             aes( x=Annual_Premium, fill=Response))+
  geom_histogram(alpha=0.4,position="identity")+
  xlab("Premio Annuale")+
  scale_y_continuous("Numero di osservazioni") +
  scale_x_continuous(limits = c(0, 110000))

ggarrange(age_plot, ggarrange(vintage_plot, premium_plot, ncol = 2, legend = "none"), nrow = 2, common.legend = T, legend = "bottom")



#############Correlazione#######################
correlationMatrix <- stats::cor(insurance[c(3, 7, 9, 11)])
corrplot(correlationMatrix, method="circle", type="lower", tl.col="black")


#############Notiamo un notevole sbilanciamento#

# Basic piechart
freq <- as.data.frame(table(insurance$Response))
colnames(freq)[1] = "Response"
freq$perc <- prop.table(freq$Freq)
ggplot(freq, aes(x = "", y = perc, fill = Response)) +
  geom_col(color = "black") +
  coord_polar(theta = "y") +
  xlab("") + 
  ylab("") +
  ggtitle("Frequenza delle classi") +
  theme(plot.title = element_text(hjust = 0.5))

#boxplot
par(mfrow=c(1,3))
boxplot(insurance$Age, main="Age")
boxplot(insurance$Annual_Premium, main="Annual_Premium")
boxplot(insurance$Vintage, main="Vintage")


#############Separazione stratificata train e test#####
set.seed(1)

train <- as.data.frame(stratified(insurance, c('Response'), 0.7))
test <- insurance[which(!(insurance$id %in% train$id)),]
insurance <- insurance[, -1]
train <- train[, -1]
test <- test[,-1]

#############Albero di decisione################
tree <- rpart(Response~., train, method = "class", control = rpart.control(cp = 0.00001))
plot(tree)
prediction <- predict(tree, test, type = "class")
confusionMatrix(as.factor(prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(prediction))
print(auc)
plot(auc)


########BILANCIAMENTO#############################
########Rimozione casuale
set.seed(1)
rem <- ubUnder(train[,1:11], train$Response, perc=14, method = "percUnder")
casual.balance.train <- data.frame(rem$X, rem$Y)
casual.balance.train$rem.Y <- NULL
names(casual.balance.train)[11] <- "Response"
#casual.balance.test <- c(test, train[rem$id.rm,])
rm(rem)
table(casual.balance.train$Response)


########SMOTE
set.seed(1)
SMOTE <- ubBalance(train[,-c(1,11)], train[,11],type="ubSMOTE", positive = 1, percOver = 100, percUnder = 280, verbose = T)
SMOTE_ds <- SMOTE$X
SMOTE_ds$Response <- SMOTE$Y
#smote.balance.test <- c(test, SMOTE$id.rm)
table(SMOTE_ds$Response)

#########Alberi di decisione#####################
#######Rimozione casuale:
tree <- rpart(Response~., casual.balance.train, method = "class", control = rpart.control(cp = 0.00001))
plot(tree)
fancyRpartPlot(tree)
prediction <- predict(tree, test, type = "class")
confusionMatrix(as.factor(prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(prediction))
print(auc)
plot(auc)
#######SMOTE:
tree <- rpart(Response~., SMOTE_ds, method = "class", control = rpart.control(cp = 0.0001))
plot(tree)
fancyRpartPlot(tree)
prediction <- predict(tree, test, type = "class")
confusionMatrix(as.factor(prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(prediction))
print(auc)
plot(auc)

#########NAIVE BAYES##################################
########SBILANCIATO:
bayes <- naiveBayes(Response ~ ., data = train)
b_prediction <- predict(bayes, newdata = test)
confusionMatrix(as.factor(b_prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(b_prediction))
print(auc)
plot(auc)

############BILANCIATO:
####CON UNDERSAMPLING RANDOM
bayes_bil <- naiveBayes(Response ~ ., data = casual.balance.train)
b_prediction <- predict(bayes_bil, newdata = test)
confusionMatrix(as.factor(b_prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(b_prediction))
print(auc)
plot(auc)

####CON SMOTE
bayes_bil <- naiveBayes(Response ~ ., data = SMOTE_ds)
b_prediction <- predict(bayes_bil, newdata = test)
confusionMatrix(as.factor(b_prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(b_prediction))
print(auc)
plot(auc)


###########SVM con dati sbilanciati#####################
#con pesi ma sbilanciato
weights <- c(1,5)
names(weights) <- c(0, 1)
support <- svm(formula = Response ~ ., data = train[, 1:11], kernel = "linear", class.weights = weights, scale = TRUE)
print(support)
plot(support, casual.balance.train)
svm_prediction <- predict(support, newdata = test)
confusionMatrix(as.factor(svm_prediction),as.factor(test$Response))
auc = roc(test$Response, as.numeric(svm_prediction))
print(auc)
plot(auc)

############BOOSTING####################################
#boosting(Response ~ ., data = casual.balance.train)
#su dataset sbilanciato
library(fastAdaboost)
ab <- adaboost(Response ~ ., data = train[,1:11], nIter=100)
boosting_prediction <- predict(ab, newdata = test)
confusionMatrix(boosting_prediction$class,as.factor(test$Response))

ab_casual.balance <- adaboost(Response ~ ., data = casual.balance.train[,1:11], nIter=100)
boosting.balanced.prediction <- predict(ab_casual.balance, newdata = test)
confusionMatrix(boosting.balanced.prediction$class,as.factor(test$Response))

#BOOSTING CON XGBOOST
library(xgboost)
library(Matrix)

xgb_preprocess <- function(df)
{
  df.xgb <- train[,c(1,2,3,5,6,7,8,10,11)]
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
weight <- ifelse(train$Response==1, 6,1)
bst <- xgboost(data = train.xgb[,-9], label = train.xgb[,9], nthread = -1, nrounds = 100, weight = weight)

test.xgb = xgb_preprocess(test)
test.xgb <- test.xgb[,-9]

prediction <- predict(bst, test.xgb)
prediction <- as.numeric(prediction > 0.5)
confusionMatrix(as.factor(prediction),as.factor(test$Response))
  
