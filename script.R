##### SETUP #####

# Download and import data sets

if(!file.exists("./data/pml-training.csv"))
        download.file(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                "./data/pml-training.csv")
if(!file.exists("./data/pml-testing.csv"))
        download.file(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                "./data/pml-testing.csv")
training <- read.csv("./data/pml-training.csv",
                        na.strings = c("","NA","#DIV/0!"))
testing <- read.csv("./data/pml-testing.csv",
                       na.strings = c("","NA","#DIV/0!"))

# Load relevant libraries

library(caret)
library(parallel)
library(doParallel)

# Configure parallel processing cluster

cluster <- makeCluster(detectCores()-1) # convention to leave 1 core for OS
registerDoParallel(cluster)


##### DATA CLEANING #####

# Remove label/junk variables

training.red1 <- training[,-c(1:7)]

# Remove variables with mostly missing values

nas <- sort(sapply(training.red1, function(x) sum(is.na(x))), decreasing = TRUE)
unique(nas) # either >=19216 (98% of records) or 0 missing values by variable
training.red2 <- training.red1[,names(nas[nas == 0])] # cull sparse variables

# Applying final column selection to test data set
cols <- colnames(training.red2)[-53] # excluding "classe" from column names
testing.red2 <- testing[,cols]


##### MODEL BUILD #####

# Preprocessing: apply PCA to further compress data (training and test)

preproc <- preProcess(training.red2[,-53], method = "pca", thresh = 0.975)
training.pca <- predict(preproc, training.red2[,-53])
training.pca <- cbind(classe = training.red2$classe, training.pca)

testing.pca <- predict(preproc, testing.red2)

# Configure model fitting: parallel processing cluster and cross-validation

fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Fit random forest model (with measuring running time)

start_time <- Sys.time()
fit <- train(classe ~ ., method="rf", data = training.pca,
             trControl = fitControl)
end_time <- Sys.time()
runtime <- end_time-start_time # approx. 7 minutes

# Evaluate model

fit
fit$resample
confusionMatrix.train(fit)

# Apply same preprocessing to test data and predict on test data

testing.pca <- predict(preproc, testing.red2)
prediction <- predict(fit, testing.pca)
data.frame(problem_id = testing$problem_id, prediction)


##### CLEAN-UP #####

# De-register parallel processing cluster

stopCluster(cluster)
registerDoSEQ()