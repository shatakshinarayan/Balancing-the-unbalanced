## Section 3: Sampling techniques:
# When the target variable's class of interest is very infrequent or rare, 
# sampling on the rare group to balance the data helps building unbiased models.
# We have demonstrated five sampling techniques and compared them with the help 
# of results from decision tree model, to identify the best technique in this case.

# Sampling Technique 1 - over sampling
overSample = ovun.sample(Class~., 
                         data = trainsplit, 
                         method = "over", 
                         N= 284328, seed =1 )$data
table(overSample$Class)
# Output
# 0       1 
# 199021  85307 
table(trainsplit$Class)
# Output
#  1      0 
# 345  199021 

#  Sampling Technique 2 - Undersampling
underSample = ovun.sample(Class~., 
                          data = trainsplit, 
                          method = "under", 
                          N= 478, seed =1 )$data
table(underSample$Class)
# Output
#  0    1 
# 133  345 

# Sampling Technique 3 - Both
bothSample =ovun.sample(Class~., 
                        data = trainsplit, 
                        method = "both",
                        N= 284807, seed =1 )$data
table(bothSample$Class)
# Output
#   0      1 
# 142149 142658 

#  Sampling Technique 4 - ROSE
data.rose <- ROSE(Class ~ ., data = trainsplit, seed =1)$data
table(data.rose$Class)
# Output
#   0     1 
# 99527  99839


# Sampling Technique 5 - SMOTE Creating SMOTE dataset with new train and test split
set.seed(4356)
smote_data <- SMOTE(Class ~ ., 
                    data  = trainsplit,
                    perc.over = 300,
                    perc.under = 150, k=3)
ggplot(smote_data,aes(x = smote_data$Class, fill="red")) + 
  geom_bar(position = "dodge", alpha = 0.5, col ="black") +
  scale_x_discrete( name = "Is it Fraud?") +
  scale_y_continuous() + 
  ggtitle("Fraud Case Classes") +
  theme(plot.title = element_text(hjust = 0.5))
table(smote_data$Class)
# Output
#  1    0 
# 1380 1552 

# Comparison of techniques on one model:

# build decision tree models
#Rose sample
t.control <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = TRUE)

cv.tree.rose <- train(Class ~ ., data = data.rose, 
                      trControl = t.control, 
                      method = "rpart", 
                      tuneLength=5)

data_rose_pred = predict(cv.tree.rose,testSplit)
confusionMatrix(data_rose_pred, testSplit$Class)
roc.curve(data_rose_pred, testSplit$Class, plotit = T)

#Confusion Matrix and Statistics

#Reference
#Prediction     1     0
#1   127  1981
#0    20 83313

#Accuracy : 0.9766                                                    
#Sensitivity : 0.863946       
#Specificity : 0.976774       

#Area under the curve (AUC): 0.530

#OverSample
t.control <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = TRUE)

cv.tree.overSample <- train(Class ~ ., data = overSample, 
                            trControl = t.control, 
                            method = "rpart", 
                            tuneLength=5)

data_overSample_pred = predict(cv.tree.overSample,testSplit)
confusionMatrix(data_overSample_pred, testSplit$Class)
roc.curve(data_overSample_pred, testSplit$Class, plotit = T)

#Confusion Matrix and Statistics

#Reference
#Prediction     1     0
#1   122   294
#0    25 85000

#Accuracy : 0.9963  

#Sensitivity : 0.829932        
#Specificity : 0.996553   

#Area under the curve (AUC): 0.646

#underSample
t.control <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = TRUE)

cv.tree.underSample <- train(Class ~ ., data = underSample, 
                             trControl = t.control, 
                             method = "rpart", 
                             tuneLength=5)

data_underSample_pred = predict(cv.tree.underSample,testsplit)
confusionMatrix(data_underSample_pred, testsplit$Class)
roc.curve(data_underSample_pred, testsplit$Class, plotit = T)

#Confusion Matrix and Statistics
#Reference
#Prediction     1     0
#1   137 16608
#0    10 68686

#Accuracy : 0.8055 
#Sensitivity : 0.931973        
#Specificity : 0.805285 

#Area under the curve (AUC): 0.504

#Hybrid Sampling
t.control <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = TRUE)

cv.tree.bothSample <- train(Class ~ ., data = bothSample, 
                            trControl = t.control, 
                            method = "rpart", 
                            tuneLength=5)

data_bothSample_pred = predict(cv.tree.bothSample,testsplit)
confusionMatrix(data_bothSample_pred, testsplit$Class)
roc.curve(data_bothSample_pred, testsplit$Class, plotit = T)

#Reference
#Prediction     1     0
#1   132  3917
#0    15 81377

#Accuracy : 0.954   

#Sensitivity : 0.897959        
#Specificity : 0.954076 

#Area under the curve (AUC): 0.516

#Smote data
t.control <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = TRUE)

cv.tree.smote <- train(Class ~ ., data = smote_data, 
                       trControl = t.control, 
                       method = "rpart", 
                       tuneLength=5)

data_smote_pred = predict(cv.tree.smote,testsplit)
confusionMatrix(data_smote_pred, testsplit$Class)
roc.curve(data_smote_pred, testsplit$Class, plotit = T)

#Reference
#Prediction
#1     0
#1   128  2828
#0    19 82466

#Accuracy : 0.9667   

#Sensitivity : 0.870748        
#Specificity : 0.966844 

#Area under the curve (AUC): 0.522

#Above results suggest SMOTE is the most favorable sampling technique based on accuracy.
