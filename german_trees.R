# load libraries for the work, comments indicate what each is for
suppressPackageStartupMessages({
    library(dplyr) # data wrangling
    library(data.table) # for handling evaluation logs from xgboost
    library(tidyr) # reshaping data sets
    library(purrr) # functional programming package
    library(corrplot) # easy plotting of correlation matrix
    library(caret) # machine learning
    library(xgboost) # machine learning
    library(Matrix) # for creating model matrices
    library(readr) # for reading data direct to tibbles
    library(ggplot2) # data viz
    library(viridis) # colour palettes for plots
    library(stringi) # string manipulation
    library(pROC) # handling ROC objects
    library(plotROC) # plotting ROC curves nicely in ggplot2
    library(forcats) # managing factors/categorical variables
    library(knitr) # pretty printing of tables with kable()
    library(rattle) # pretty printing of decision tree
    library(rBayesianOptimization)
    source("./R/print_conf_matrix.R")
})

# url for the main dataset
data_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
# the data dictionary
data_dict_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"

# local filepaths
data_path <- paste0("./data/", basename(data_url))
dict_path <- paste0("./data/", basename(data_dict_url))

# download files if not already present
if (!file.exists(data_path)) {
    download.file(data_url, data_path)
}
if (!file.exists(dict_path)) {
    download.file(data_dict_url, dict_path)
}

# column names taken from the data dictionary, slightly changed to keep lengths 
# reasonable
column_names <- c("acct_status", "duration", "credit_history", 
                  "purpose", "amount", "savings_acct", 
                  "present_emp_since", "pct_of_income", "sex_status", 
                  "other_debtor_guarantor", "resident_since", "property", 
                  "age", "other_debts", "housing", 
                  "num_existing_credits", "job", "num_dependents", 
                  "telephone", "foreign_worker", "outcome")
# set the column types manually to avoid any coercion errors
column_types <- c("ciccicciccicicciciccc")

raw_df <- read_delim(data_path, 
                     delim = " ", 
                     col_names = column_names, 
                     col_types = column_types)

# details of the factors are taken from the data dictionary
# Can ignore warnings, which are because there are no rows with the given 
# Axx code. These manipulations use functions from the forcats package, 
# which makes factors much easier
clean_df <- raw_df %>% 
    mutate(acct_status = fct_recode(acct_status, 
                                    overdrawn = "A11", below_200DM = "A12", 
                                    over_200DM = "A13", no_acct = "A14"), 
           credit_history = fct_recode(credit_history, 
                                       none_taken_all_paid = "A30", 
                                       all_paid_this_bank = "A31", 
                                       all_paid_duly = "A32", 
                                       past_delays = "A33", 
                                       critical_acct = "A34"), 
           purpose = fct_recode(purpose, 
                                car_new = "A40", car_used = "A41", 
                                furniture_equipment  = "A42", 
                                radio_tv = "A43",  
                                dom_appliance = "A44", 
                                repairs = "A45", 
                                education = "A46", 
                                vacation = "A47", 
                                retraining = "A48", 
                                business = "A49",  
                                others = "A410"), 
           savings_acct = fct_recode(savings_acct, 
                                     to_100DM = "A61", 
                                     to_500DM = "A62", 
                                     to_1000DM = "A63", 
                                     over_1000DM = "A64", 
                                     unknwn_no_acct = "A65"),
           present_emp_since = fct_recode(present_emp_since, 
                                          unemployed = "A71", 
                                          to_1_yr = "A72", 
                                          to_4_yrs = "A73", 
                                          to_7_yrs = "A74", 
                                          over_7_yrs = "A75"), 
           sex_status = fct_recode(sex_status, 
                                   male_divorced = "A91", 
                                   female_married = "A92", 
                                   male_single = "A93",  
                                   male_married = "A94", 
                                   female_single = "A95"), 
           other_debtor_guarantor = fct_recode(other_debtor_guarantor, 
                                               none = "A101",  
                                               co_applicant = "A102", 
                                               guarantor = "A103"), 
           property = fct_recode(property, 
                                 real_estate = "A121", 
                                 savings_insurance = "A122", 
                                 car_other = "A123", 
                                 unknwn_none = "A124"), 
           other_debts = fct_recode(other_debts, 
                                    bank = "A141", 
                                    stores = "A142", 
                                    none = "A143"), 
           housing = fct_recode(housing, 
                                rent = "A151", 
                                own = "A152", 
                                for_free = "A153"), 
           job = fct_recode(job, 
                            unemp_unskilled_nonres = "A171", 
                            unskilled_res = "A172",  
                            skilled_official = "A173", 
                            mgmt_highqual = "A174"), 
           telephone = fct_recode(telephone, 
                                  no = "A191", 
                                  yes = "A192"), 
           foreign_worker = fct_recode(foreign_worker, 
                                       yes = "A201", 
                                       no = "A202"), 
           outcome = fct_recode(outcome, 
                                good = "1", 
                                bad = "2")) %>% 
    # add another factor for gender, a simplification of sex_status, which can 
    # then be compared during EDA
    mutate(gender = fct_collapse(sex_status, 
                                 male = "male_divorced", 
                                 male = "male_single", 
                                 male = "male_married", 
                                 female = "female_married", 
                                 female = "female_single"))

glimpse(clean_df)

factor_varnames <- colnames(clean_df)[sapply(clean_df, is.factor)]
cat_pct <- 0.02
walk(factor_varnames, function(x) {
    tmp_tbl <- fct_count(clean_df[[x]]) %>% 
        mutate(pct = n/sum(n))
    if (min(tmp_tbl[["pct"]]) < cat_pct) {
        print(paste0(x, " has categories with less than ", 
                     as.integer(100 * cat_pct), "% of observations."))
        print(table(clean_df[[x]])/nrow(clean_df))
    }
})


# start with bar plots for the factor vars
clean_df %>% 
    dplyr::select(-outcome) %>% 
    select_if(is.factor) %>% 
    mutate_all(as.character) %>% # this is to avoid an error about factor 
    # attributes being lost, but this step isn't 
    # strictly necessary as gather() would coerce 
    # to character anyway
    gather(key = "feature") %>% # gather() turns a wide table to a long one
    ggplot(aes(x = value)) + 
    geom_bar() + 
    facet_wrap(~ feature, scales = "free_x") + 
    theme_minimal() + 
    theme(axis.text.x = element_blank())

# then histograms for integers
clean_df %>% 
    dplyr::select_if(is.integer) %>% 
    gather(key = "feature") %>% 
    ggplot(aes(x = value)) +
    geom_histogram(bins = 20) +
    facet_wrap(~ feature, scales = "free_x") + 
    theme_minimal() +
    theme(axis.text.x = element_blank())


walk(c("foreign_worker", "other_debtor_guarantor", "sex_status"), 
     function(x) {
         tmp_tbl <- fct_count(clean_df[[x]]) %>% 
             mutate(pct = n/sum(n))
         # print(paste0(x, " has categories with less than ", 
         #              as.integer(100 * cat_pct), "% of observations."))
         print(paste0("Proportions for variable: ", x))
         print(table(clean_df[[x]])/nrow(clean_df))
     })


clean_df %>% 
    select(sex_status, gender) %>% 
    mutate_all(as.character) %>% 
    gather(key = "feature") %>% 
    ggplot(aes(x = value)) + 
    geom_bar() + 
    facet_wrap(~ feature, scales = "free_x") + 
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

clean_df$sex_status <- NULL


# set a seed to make analysis reproducible
set.seed(1705L)
# caret::createDataPartition() will ensure that the overall proportions are 
# reflected in each partition
in_train <- createDataPartition(clean_df[["outcome"]], 
                                p = 0.8, 
                                list = FALSE)
train_df <- clean_df[in_train, ]
test_df <- clean_df[-in_train, ]


corr_matrix <- train_df %>% 
    select_if(is.integer) %>% 
    data.matrix() %>% 
    cor()
corrplot(corr_matrix)


cor(train_df$amount, train_df$duration)


# set up simple logistic regression for each of the variables with outcome
amount_model <- glm(outcome ~ amount, 
                    family = binomial(link = "logit"), 
                    data = train_df)
duration_model <- glm(outcome ~ duration, 
                      family = binomial(link = "logit"), 
                      data = train_df)
# examine the summary of residuals
summary(amount_model$residuals)
summary(duration_model$residuals)


rpart_trcontrol <- trainControl(method = "repeatedcv", 
								number = 5L, 
								repeats = 3L, 
								summaryFunction = twoClassSummary, 
								classProbs = TRUE, 
								savePredictions = "final")

if (file.exists("./data/rpart_model.rds")) {
	rpart_model <- read_rds("./data/rpart_model.rds")
} else {
	rpart_model <- rpart_model <- train(outcome ~ ., 
										data = train_df, 
										method = "rpart", 
										metric = "ROC", 
										tuneLength = 30, 
										trControl = rpart_trcontrol)
	
	write_rds(rpart_model, "./data/rpart_model.rds")
}
fancyRpartPlot(rpart_model$finalModel, sub = "German Credit Decision Tree")


rpart_train_probs <- predict(rpart_model, train_df, type = "prob")[["good"]]
rpart_train_roc <- roc(train_df$outcome, rpart_train_probs)
rpart_best_threshold <- round(coords(rpart_train_roc, 
									 x = "best", 
									 ret = "threshold", 
									 best.method = "closest.topleft"), 
							  2)
plot(rpart_train_roc, type = "s", print.thres = "best", print.thres.best.method = "closest.topleft", print.auc = TRUE)


rpart_preds <- as.character(
	predict(rpart_model, 
			test_df, 
			type = "prob")[["good"]] > rpart_best_threshold) %>% 
	fct_recode(good = "TRUE", 
               bad = "FALSE")
rpart_conf_mat <- confusionMatrix(rpart_preds, test_df$outcome)
print_conf_matrix(rpart_conf_mat)
rpart_roc <- roc(as.numeric(test_df$outcome == "good"), 
				 as.numeric(rpart_preds == "good"))
auc(rpart_roc)


plot(varImp(rpart_model), top = 10)


# set up trainControl object to pass to caret::train()
rf_trcontrol <- trainControl(method = "repeatedcv", 
                             # 5-fold cross-validationm repeated 3 times
                             number = 5L, 
                             repeats = 3L,
                             summaryFunction = twoClassSummary, 
                             search = "random", 
                             classProbs = TRUE, 
                             savePredictions = "final")
# set up tuning grid for testing values of mtry
rf_tunegrid <- expand.grid(.mtry = seq(2, 20, by = 1))

if (file.exists("./data/rf_model.rds")) {
    rf_model <- read_rds("./data/rf_model.rds")
} else {
    rf_model <- train(outcome ~ ., 
                      data = train_df, 
                      method = "rf", 
                      metric = "ROC", 
                      maximize = TRUE, 
                      tuneGrid = rf_tunegrid,
                      nodesize = 20L, 
                      ntree = 1000L, 
                      trControl = rf_trcontrol)
    
    write_rds(rf_model, "./data/rf_model.rds")
}
print(rf_model)
plot(rf_model)


rf_train_probs <- predict(rf_model, train_df, type = "prob")[["good"]]
rf_train_roc <- roc(train_df$outcome, rf_train_probs)
rf_best_threshold <- round(coords(rf_train_roc, 
								  x = "best", 
								  ret = "threshold", 
								  best.method = "closest.topleft"), 
						   2)
plot(rf_train_roc, 
	 type = "s", 
	 print.thres = "best", 
	 print.thres.best.method = "closest.topleft", 
	 print.auc = TRUE)
paste0("Threshold that maximises AUC is ", 
	   rf_best_threshold)


rf_preds <- as.character(
	predict(rf_model, 
			test_df, 
			type = "prob")[["good"]] > rf_best_threshold) %>% 
	fct_recode(good = "TRUE", 
               bad = "FALSE")
rf_conf_mat <- confusionMatrix(rf_preds, test_df$outcome)
print_conf_matrix(rf_conf_mat)
rf_roc <- roc(as.numeric(test_df$outcome == "good"), 
			  as.numeric(rf_preds == "good"))
auc(rf_roc)


plot(varImp(rf_model), top = 10)


# create matrices with Matrix::sparse.model.matrix 
train_sparse <- sparse.model.matrix(outcome ~ ., data = train_df)
test_sparse <- sparse.model.matrix(outcome ~ ., data = test_df)
# xgboost needs binary numeric labels
train_labels <- if_else(train_df$outcome == "good", 1L, 0L)
test_labels <- if_else(test_df$outcome == "good", 1L, 0L)

# xgb performs best with its own object type
dtrain <- xgb.DMatrix(train_sparse, label = train_labels)
dtest <- xgb.DMatrix(test_sparse, label = test_labels)

# set seed for reproducibility
set.seed(1907L)

# define the folds for cross-validation
cv_folds <- KFold(train_labels, nfolds = 5,
                  stratified = TRUE, seed = 0)

# create a function that will return values for the optimisation
xgb_cv_bayes <- function(max_depth, 
                         min_child_weight, 
                         subsample, 
                         eta,
                         colsample_bytree, 
                         lambda, 
                         alpha, 
                         gamma) {
    cv <- xgb.cv(params = list(booster = "gbtree", 
                               eta = eta,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               subsample = subsample, 
                               colsample_bytree = colsample_bytree,
                               lambda = lambda, 
                               alpha = alpha,
                               gamma = gamma, 
                               objective = "binary:logistic",
                               eval_metric = "auc"),
                 data = dtrain, 
                 nrounds = 10000L,
                 folds = cv_folds, 
                 prediction = TRUE, 
                 showsd = TRUE,
                 early_stopping_rounds = 100L, 
                 maximize = TRUE, 
                 verbose = 0)
    list(Score = cv$evaluation_log$test_auc_mean[cv$best_iteration],
         Pred = cv$pred)
}

# run optimisation with bounds for the parameters to be tested
if (file.exists("./data/xgb_opt_res.rds")) {
    xgb_opt_res <- read_rds("./data/xgb_opt_res.rds")
} else {
    xgb_opt_res <- BayesianOptimization(xgb_cv_bayes,
                                        bounds = list(
                                            max_depth = c(1L, 15L),
                                            min_child_weight = c(1L, 20L), 
                                            subsample = c(0.2, 1.0), 
                                            eta = c(0.0001, 0.1), 
                                            colsample_bytree = c(0.2, 1), 
                                            lambda = c(0.5, 20.0), 
                                            alpha = c(0.5, 20.0), 
                                            gamma = c(0.0, 20.0)),
                                        init_grid_dt = NULL, 
                                        init_points = 10, 
                                        # number of random points used to 
                                        # initialise the process. Must be at 
                                        # least 1 or will fail
                                        n_iter = 50, # this is a lot of rounds 
                                        # and will take quite some time to run. 
                                        # Can get good results n_iter = 20.
                                        acq = "ucb", 
                                        kappa = 2.576, 
                                        eps = 0.0,
                                        verbose = TRUE)
    write_rds(xgb_opt_res, "./data/xgb_opt_res.rds")
}
xgb_opt_res$Best_Par


xgb_params <- xgb_opt_res[["Best_Par"]]
xgb_param_list <- list(
    eta = xgb_params[["eta"]], 
    max_depth = xgb_params[["max_depth"]], 
    min_child_weight = xgb_params[["min_child_weight"]], 
    subsample = xgb_params[["subsample"]], 
    colsample_bytree = xgb_params[["colsample_bytree"]], 
    lambda = xgb_params[["lambda"]], 
    alpha = xgb_params[["alpha"]], 
    gamma = xgb_params[["gamma"]], 
    objective = "binary:logistic", 
    eval_metric = "auc")

if (file.exists("./data/xgb_cv.rds")) {
    xgb_cv <- read_rds("./data/xgb_cv.rds")
} else {
    xgb_cv <- xgb.cv(params = xgb_param_list, 
                     data = dtrain, 
                     nrounds = 10000L,
                     folds = cv_folds, 
                     prediction = TRUE, 
                     showsd = TRUE,
                     early_stopping_rounds = 1000L, 
                     print_every_n = 500L, 
                     maximize = TRUE)
    
    write_rds(xgb_cv, "./data/xgb_cv.rds")
}


xgb_cv[["evaluation_log"]] %>% 
    dplyr::select(iter, train_auc_mean, test_auc_mean) %>% 
    gather(key = "partition", 
           value = "auc", 
           train_auc_mean, 
           test_auc_mean) %>% 
    mutate(partition = stri_extract_first_regex(partition, "^[a-z]+")) %>% 
    ggplot(aes(x = iter, y = auc, colour = partition)) +
    geom_point() +
    scale_colour_viridis(discrete = TRUE, option = "B") + 
    ylim(0, 1) +
    theme_minimal()


xgb_model <- xgb.train(params = xgb_param_list, 
                       data = dtrain, 
                       nrounds = xgb_cv[["best_iteration"]], 
                       metrics = "auc")


var_importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(var_importance, top = 10)


xgb_train_probs <- predict(xgb_model, dtrain, type = "prob")
xgb_train_roc <- roc(train_labels, xgb_train_probs)
xgb_best_threshold <- round(coords(xgb_train_roc, 
								  x = "best", 
								  ret = "threshold", 
								  best.method = "closest.topleft"), 
						   2)
plot(xgb_train_roc, 
	 type = "s", 
	 print.thres = "best", 
	 print.thres.best.method = "closest.topleft", 
	 print.auc = TRUE)
paste0("Threshold that maximises AUC is ", 
	   xgb_best_threshold)


xgb_preds <- predict(xgb_model, dtest, type = "prob")
xgb_factor_preds <- as.character(xgb_preds > xgb_best_threshold) %>% 
    fct_recode(good = "TRUE", 
               bad = "FALSE")
xgb_conf_mat <- confusionMatrix(xgb_factor_preds, test_df$outcome)
print_conf_matrix(xgb_conf_mat)


xgb_roc <- roc(test_labels, as.numeric(xgb_factor_preds))
auc(xgb_roc)


tibble(model = c("Decision Tree", 
				 "Random Forest", 
				 "XGBoost"), 
	   AUC = map_dbl(list(rpart_roc, rf_roc, xgb_roc), function(roc_obj) {
	   	round(as.numeric(auc(roc_obj)), 2)
	   }
	   )) %>% 
	ggplot(aes(x = factor(model), y = AUC, fill = model, label = AUC)) + 
	geom_col(color = "light grey") + 
	labs(y = "AUC", x = "") + 
	scale_fill_viridis(option = "B", discrete = TRUE) + 
	scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.2)) +
	geom_hline(yintercept = 1, col = inferno(9)[5]) + 
	guides(fill = FALSE) + 
	theme_minimal() + 
	theme(axis.text.x = element_text(size = 14)) + 
	geom_text(nudge_y = 0.05, size = 5)


names(clean_df)


# create matrices with Matrix::sparse.model.matrix as before, but with 
# extra step of dropping age and gender
train_df2 <- train_df %>% 
	dplyr::select(-age, -gender)
test_df2 <- test_df %>% 
	dplyr::select(-age, -gender)
train_sparse2 <- sparse.model.matrix(outcome ~ ., data = train_df2)
test_sparse2 <- sparse.model.matrix(outcome ~ ., data = test_df2)

# xgb performs best with its own object type
dtrain2 <- xgb.DMatrix(train_sparse2, label = train_labels)
dtest2 <- xgb.DMatrix(test_sparse2, label = test_labels)

# set seed for reproducibility
set.seed(1907L)

# define the folds for cross-validation
cv_folds <- KFold(train_labels, nfolds = 5,
                  stratified = TRUE, seed = 0)

# create a function that will return values for the optimisation
xgb_cv_bayes <- function(max_depth, 
                         min_child_weight, 
                         subsample, 
                         eta,
                         colsample_bytree, 
                         lambda, 
                         alpha, 
                         gamma) {
    cv <- xgb.cv(params = list(booster = "gbtree", 
                               eta = eta,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               subsample = subsample, 
                               colsample_bytree = colsample_bytree,
                               lambda = lambda, 
                               alpha = alpha,
                               gamma = gamma, 
                               objective = "binary:logistic",
                               eval_metric = "auc"),
                 data = dtrain2, 
                 nrounds = 10000L,
                 folds = cv_folds, 
                 prediction = TRUE, 
                 showsd = TRUE,
                 early_stopping_rounds = 100L, 
                 maximize = TRUE, 
                 verbose = 0)
    list(Score = cv$evaluation_log$test_auc_mean[cv$best_iteration],
         Pred = cv$pred)
}

# run optimisation with bounds for the parameters to be tested
if (file.exists("./data/xgb_opt_res2.rds")) {
    xgb_opt_res2 <- read_rds("./data/xgb_opt_res2.rds")
} else {
    xgb_opt_res2 <- BayesianOptimization(xgb_cv_bayes,
                                        bounds = list(
                                            max_depth = c(1L, 15L),
                                            min_child_weight = c(1L, 20L), 
                                            subsample = c(0.2, 1.0), 
                                            eta = c(0.0001, 0.1), 
                                            colsample_bytree = c(0.2, 1), 
                                            lambda = c(0.5, 20.0), 
                                            alpha = c(0.5, 20.0), 
                                            gamma = c(0.0, 20.0)),
                                        init_grid_dt = NULL, 
                                        init_points = 10, 
                                        # number of random points used to 
                                        # initialise the process. Must be at 
                                        # least 1 or will fail
                                        n_iter = 50, # this is a lot of rounds 
                                        # and will take quite some time to run. 
                                        # Can get good results n_iter = 20.
                                        acq = "ucb", 
                                        kappa = 2.576, 
                                        eps = 0.0,
                                        verbose = TRUE)
    write_rds(xgb_opt_res2, "./data/xgb_opt_res2.rds")
}
xgb_opt_res2$Best_Par


xgb_params2 <- xgb_opt_res2[["Best_Par"]]
xgb_param_list2 <- list(
    eta = xgb_params2[["eta"]], 
    max_depth = xgb_params2[["max_depth"]], 
    min_child_weight = xgb_params2[["min_child_weight"]], 
    subsample = xgb_params2[["subsample"]], 
    colsample_bytree = xgb_params2[["colsample_bytree"]], 
    lambda = xgb_params2[["lambda"]], 
    alpha = xgb_params2[["alpha"]], 
    gamma = xgb_params2[["gamma"]], 
    objective = "binary:logistic", 
    eval_metric = "auc")

if (file.exists("./data/xgb_cv2.rds")) {
    xgb_cv2 <- read_rds("./data/xgb_cv2.rds")
} else {
    xgb_cv2 <- xgb.cv(params = xgb_param_list2, 
                     data = dtrain2, 
                     nrounds = 10000L,
                     folds = cv_folds, 
                     prediction = TRUE, 
                     showsd = TRUE,
                     early_stopping_rounds = 1000L, 
                     print_every_n = 500L, 
                     maximize = TRUE)
    
    write_rds(xgb_cv2, "./data/xgb_cv2.rds")
}


xgb_cv2[["evaluation_log"]] %>% 
    dplyr::select(iter, train_auc_mean, test_auc_mean) %>% 
    gather(key = "partition", 
           value = "auc", 
           train_auc_mean, 
           test_auc_mean) %>% 
    mutate(partition = stri_extract_first_regex(partition, "^[a-z]+")) %>% 
    ggplot(aes(x = iter, y = auc, colour = partition)) +
    geom_point() +
    scale_colour_viridis(discrete = TRUE, option = "B") + 
    ylim(0, 1) +
    theme_minimal()


xgb_model2 <- xgb.train(params = xgb_param_list2, 
                       data = dtrain2, 
                       nrounds = xgb_cv2[["best_iteration"]], 
                       metrics = "auc")


var_importance <- xgb.importance(model = xgb_model2)
xgb.plot.importance(var_importance, top = 10)


xgb_train_probs2 <- predict(xgb_model2, dtrain2, type = "prob")
xgb_train_roc2 <- roc(train_labels, xgb_train_probs2)
xgb_best_threshold2 <- round(coords(xgb_train_roc2, 
									x = "best", 
									ret = "threshold", 
									best.method = "closest.topleft"), 
							 2)
plot(xgb_train_roc2, 
	 type = "s", 
	 print.thres = "best", 
	 print.thres.best.method = "closest.topleft", 
	 print.auc = TRUE)
paste0("Threshold that maximises AUC is ", 
	   xgb_best_threshold2)


xgb_preds2 <- predict(xgb_model2, dtest2, type = "prob")
xgb_factor_preds2 <- as.character(xgb_preds2 > xgb_best_threshold2) %>% 
	fct_recode(good = "TRUE", 
			   bad = "FALSE")
xgb_conf_mat2 <- confusionMatrix(xgb_factor_preds2, test_df2$outcome)
print_conf_matrix(xgb_conf_mat2)


xgb_roc2 <- roc(test_labels, as.numeric(xgb_factor_preds2))
auc(xgb_roc2)


# create matrices with Matrix::sparse.model.matrix as before, but with 
# extra step of dropping age and gender
train_df3 <- train_df %>% 
	dplyr::select(-gender)
test_df3 <- test_df %>% 
	dplyr::select(-gender)
train_sparse3 <- sparse.model.matrix(outcome ~ ., data = train_df3)
test_sparse3 <- sparse.model.matrix(outcome ~ ., data = test_df3)

# xgb performs best with its own object type
dtrain3 <- xgb.DMatrix(train_sparse3, label = train_labels)
dtest3 <- xgb.DMatrix(test_sparse3, label = test_labels)

# set seed for reproducibility
set.seed(1907L)

# define the folds for cross-validation
cv_folds <- KFold(train_labels, nfolds = 5,
                  stratified = TRUE, seed = 0)

# create a function that will return values for the optimisation
xgb_cv_bayes <- function(max_depth, 
                         min_child_weight, 
                         subsample, 
                         eta,
                         colsample_bytree, 
                         lambda, 
                         alpha, 
                         gamma) {
    cv <- xgb.cv(params = list(booster = "gbtree", 
                               eta = eta,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               subsample = subsample, 
                               colsample_bytree = colsample_bytree,
                               lambda = lambda, 
                               alpha = alpha,
                               gamma = gamma, 
                               objective = "binary:logistic",
                               eval_metric = "auc"),
                 data = dtrain3, 
                 nrounds = 10000L,
                 folds = cv_folds, 
                 prediction = TRUE, 
                 showsd = TRUE,
                 early_stopping_rounds = 100L, 
                 maximize = TRUE, 
                 verbose = 0)
    list(Score = cv$evaluation_log$test_auc_mean[cv$best_iteration],
         Pred = cv$pred)
}

# run optimisation with bounds for the parameters to be tested
if (file.exists("./data/xgb_opt_res3.rds")) {
    xgb_opt_res3 <- read_rds("./data/xgb_opt_res3.rds")
} else {
    xgb_opt_res3 <- BayesianOptimization(xgb_cv_bayes,
                                        bounds = list(
                                            max_depth = c(1L, 15L),
                                            min_child_weight = c(1L, 20L), 
                                            subsample = c(0.2, 1.0), 
                                            eta = c(0.0001, 0.1), 
                                            colsample_bytree = c(0.2, 1), 
                                            lambda = c(0.5, 20.0), 
                                            alpha = c(0.5, 20.0), 
                                            gamma = c(0.0, 20.0)),
                                        init_grid_dt = NULL, 
                                        init_points = 10, 
                                        # number of random points used to 
                                        # initialise the process. Must be at 
                                        # least 1 or will fail
                                        n_iter = 50, # this is a lot of rounds 
                                        # and will take quite some time to run. 
                                        # Can get good results n_iter = 20.
                                        acq = "ucb", 
                                        kappa = 2.576, 
                                        eps = 0.0,
                                        verbose = TRUE)
    write_rds(xgb_opt_res3, "./data/xgb_opt_res3.rds")
}
xgb_opt_res3$Best_Par


xgb_params3 <- xgb_opt_res3[["Best_Par"]]
xgb_param_list3 <- list(
    eta = xgb_params3[["eta"]], 
    max_depth = xgb_params3[["max_depth"]], 
    min_child_weight = xgb_params3[["min_child_weight"]], 
    subsample = xgb_params3[["subsample"]], 
    colsample_bytree = xgb_params3[["colsample_bytree"]], 
    lambda = xgb_params3[["lambda"]], 
    alpha = xgb_params3[["alpha"]], 
    gamma = xgb_params3[["gamma"]], 
    objective = "binary:logistic", 
    eval_metric = "auc")

if (file.exists("./data/xgb_cv3.rds")) {
    xgb_cv3 <- read_rds("./data/xgb_cv3.rds")
} else {
    xgb_cv3 <- xgb.cv(params = xgb_param_list3, 
                     data = dtrain3, 
                     nrounds = 10000L,
                     folds = cv_folds, 
                     prediction = TRUE, 
                     showsd = TRUE,
                     early_stopping_rounds = 1000L, 
                     print_every_n = 500L, 
                     maximize = TRUE)
    
    write_rds(xgb_cv3, "./data/xgb_cv3.rds")
}


xgb_cv3[["evaluation_log"]] %>% 
    dplyr::select(iter, train_auc_mean, test_auc_mean) %>% 
    gather(key = "partition", 
           value = "auc", 
           train_auc_mean, 
           test_auc_mean) %>% 
    mutate(partition = stri_extract_first_regex(partition, "^[a-z]+")) %>% 
    ggplot(aes(x = iter, y = auc, colour = partition)) +
    geom_point() +
    scale_colour_viridis(discrete = TRUE, option = "B") + 
    ylim(0, 1) +
    theme_minimal()


xgb_model3 <- xgb.train(params = xgb_param_list3, 
                       data = dtrain3, 
                       nrounds = xgb_cv3[["best_iteration"]], 
                       metrics = "auc")


var_importance <- xgb.importance(model = xgb_model3)
xgb.plot.importance(var_importance, top = 10)


xgb_train_probs3 <- predict(xgb_model3, dtrain3, type = "prob")
xgb_train_roc3 <- roc(train_labels, xgb_train_probs3)
xgb_best_threshold3 <- round(coords(xgb_train_roc3, 
									x = "best", 
									ret = "threshold", 
									best.method = "closest.topleft"), 
							 2)
plot(xgb_train_roc3, 
	 type = "s", 
	 print.thres = "best", 
	 print.thres.best.method = "closest.topleft", 
	 print.auc = TRUE)
paste0("Threshold that maximises AUC is ", 
	   xgb_best_threshold3)


xgb_preds3 <- predict(xgb_model3, dtest3, type = "prob")
xgb_factor_preds3 <- as.character(xgb_preds3 > xgb_best_threshold3) %>% 
	fct_recode(good = "TRUE", 
			   bad = "FALSE")
xgb_conf_mat3 <- confusionMatrix(xgb_factor_preds3, test_df3$outcome)
print_conf_matrix(xgb_conf_mat3)


xgb_roc3 <- roc(test_labels, as.numeric(xgb_factor_preds3))
auc(xgb_roc3)


tibble(model = c("Decision Tree", 
				 "Random Forest", 
				 "XGBoost, Full", 
				 "XGBoost, no Age/Gender", 
				 "XGBoost, no Gender"), 
	   AUC = map_dbl(list(rpart_roc,
	   				      rf_roc,
	   				      xgb_roc, 
	   				      xgb_roc2, 
	   				      xgb_roc3), function(roc_obj) {
	   				      	  round(as.numeric(auc(roc_obj)), 2)
	   }
	   )) %>% 
	ggplot(aes(x = factor(model), y = AUC, fill = model, label = AUC)) + 
	geom_col(color = "light grey") + 
	labs(y = "AUC", x = "") + 
	scale_fill_viridis(option = "B", discrete = TRUE) + 
	scale_y_continuous(breaks = seq(from = 0, to = 1, by = 0.2)) +
	geom_hline(yintercept = 1, col = inferno(9)[5]) + 
	guides(fill = FALSE) + 
	theme_minimal() + 
	theme(axis.text.x = element_text(size = 8)) + 
	geom_text(nudge_y = 0.05, size = 5)


# create a basic decision tree to use as a baseline
basic_rpart <- rpart::rpart(outcome ~ ., train_df, 
                            control = list(maxdepth = 2))
lift_data <- tibble(xgb1 = xgb_preds, 
                    xgb2 = xgb_preds2, 
                    xgb3 = xgb_preds3, 
                    rf = predict(rf_model, 
                                 test_df, 
                                 type = "prob")[, "good"], 
                    rpart_full = predict(rpart_model, 
                                    test_df, 
                                    type = "prob")[, "good"], 
                    rpart_basic = predict(basic_rpart, 
                                          test_df, 
                                          type = "prob")[, "good"], 
                    outcome = test_df$outcome)
caret::lift(outcome ~ xgb1 + xgb2 + xgb3 + rf + rpart_full + rpart_basic, 
            data = lift_data, 
            class = "good") %>% 
    ggplot()

