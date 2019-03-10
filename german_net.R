# start by sourcing cutdown version of the main script, which will load the 
# data and packages
source("./german_trees.R")
library(glmnet)
# generate the CV folds myself to make results reproducible
set.seed(1907)
cv_folds_glmnet <- createFolds(train_df$outcome, 
                               list = FALSE)

# try lasso, elastic, and ridge regression
lasso_all_vars <- cv.glmnet(train_sparse, 
                            train_df$outcome, 
                            family = "binomial", 
                            foldid = cv_folds_glmnet, 
                            type.measure = "deviance", 
                            alpha = 1)
elastic_all_vars <- cv.glmnet(train_sparse, 
                              train_df$outcome, 
                              family = "binomial", 
                              foldid = cv_folds_glmnet, 
                              type.measure = "deviance", 
                              alpha = 0.5)
ridge_all_vars <- cv.glmnet(train_sparse, 
                            train_df$outcome, 
                            family = "binomial", 
                            foldid = cv_folds_glmnet, 
                            type.measure = "deviance", 
                            alpha = 0)

glmnet_models <- tibble(type = c("lasso", "elastic", "ridge"), 
                        model = list(lasso_all_vars, 
                                     elastic_all_vars, 
                                     ridge_all_vars))
# generate some plots and see the coefficients
walk(glmnet_models$model, plot)
map(glmnet_models$model, coef, s = "lambda.min")
map(glmnet_models$model, coef, s = "lambda.1se")

# keep all of the results in tibbles, allows for easily working with all models 
# at once
(glmnet_train_results <- glmnet_models %>% 
        mutate(lambda = list(c("lambda.min", "lambda.1se"))) %>% 
        unnest(lambda) %>% 
        inner_join(glmnet_models, by = "type") %>% 
        mutate(train_probs = map2(model, 
                                  lambda, 
                                  ~ predict(.x, 
                                            newx = train_sparse, 
                                            s = .y, 
                                            type = "response") %>% 
                                      # need to subset as the predictions come 
                                      # out in a matrix
                                      .[, 1])) %>% 
        mutate(train_roc = map(train_probs, 
                               ~ pROC::roc(train_labels, .x))) %>% 
        mutate(train_auc = map_dbl(train_roc, ~ .x[["auc"]] %>% 
                                       as.double)) %>% 
        arrange(desc(train_auc))
)
# results from ridge seem to be best. will use the lambda.1se value but keep the 
# lambda.min just as a comparison
(glmnet_test_results <- glmnet_train_results %>% 
        filter(type == "ridge") %>% 
        mutate(test_probs = map2(model, 
                                 lambda, 
                                 ~ predict(.x, 
                                           newx = test_sparse, 
                                           s = .y, 
                                           type = "response")) %>% 
                   map(~ .x[, 1])) %>% 
        mutate(test_roc = map(test_probs, 
                               ~ pROC::roc(test_labels, .x))) %>% 
        mutate(test_auc = map_dbl(test_roc, ~ .x[["auc"]] %>% 
                                      as.double)) %>% 
        arrange(desc(test_auc)))

# compare the different results for lambdas
glmnet_test_results %>% 
    select(type, lambda, train_auc, test_auc) %>% 
    arrange(desc(test_auc))

# add ridge to the lift data tibble, 
lift_data02 <- lift_data %>% 
    bind_cols(glmnet_test_results %>% 
               filter(type == "ridge", 
                      lambda == "lambda.1se") %>% 
               select(test_probs) %>% 
                  unnest() %>% 
                  transmute(ridge_1se = 1 - test_probs))
caret::lift(outcome ~ xgb1 + rpart_basic + ridge_1se, 
            data = lift_data02,  
            class = "good") %>% 
    ggplot()
