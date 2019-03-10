# start by sourcing cutdown version of the main script, which will load the 
# data and packages
source("./german_trees.R")
library(glmnet)
# generate the CV folds myself to make results reproducible
set.seed(1907)
cv_folds_glmnet <- createFolds(train_df$outcome, 
                               list = FALSE)
# build a new model matrix with no intercept
train_sparse <- model.matrix(outcome ~ ., data = train_df)
# try lasso, elastic, and ridge regression
lasso_all_vars <- cv.glmnet(train_sparse, 
                            train_df$outcome, 
                            family = "binomial", 
                            foldid = cv_folds_glmnet, 
                            type.measure = "deviance", 
                            alpha = 0)
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
                            alpha = 1)

glmnet_models <- tibble(type = c("lasso", "elastic", "ridge"), 
                        model = list(lasso_all_vars, 
                                     elastic_all_vars, 
                                     ridge_all_vars))
walk(glmnet_models$model, plot)
map(glmnet_models$model, coef, s = "lambda.min")
map(glmnet_models$model, coef, s = "lambda.1se")
plot(lasso_all_vars)
plot(elastic_all_vars)
plot(ridge_all_vars)

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
                                      .[, 1])) %>% 
        mutate(train_roc = map(train_probs, 
                               ~ pROC::roc(train_labels, .x))) %>% 
        mutate(train_auc = map_dbl(train_roc, ~ .x[["auc"]] %>% 
                                       as.double)) %>% 
        arrange(desc(train_auc))
)
(glmnet_test_results <- glmnet_train_results %>% 
        filter(type == "lasso") %>% 
        mutate(test_probs = map2(model, 
                                 lambda, 
                                 ~ predict(.x, 
                                           newx = test_sparse, 
                                           s = .y, 
                                           type = "response")) %>% 
                   map(~ .x[, 1])) %>% 
        mutate(test_roc = map(test_probs, 
                               ~ pROC::roc(test_labels, .x))) %>% 
        mutate(test_auc = map_dbl(train_roc, ~ .x[["auc"]] %>% 
                                      as.double)) %>% 
        arrange(desc(train_auc)))


lift_data02 <- lift_data %>% 
    select(xgb1, rf, rpart_full, rpart_basic, outcome) %>% 
    bind_cols(glmnet_test_results %>% 
               filter(type == "lasso", 
                      lambda == "lambda.1se") %>% 
               select(test_probs) %>% 
                  unnest() %>% 
                  transmute(lasso_1se = 1 - test_probs))
caret::lift(outcome ~ xgb1 + rf + rpart_full + rpart_basic + lasso_1se, 
            data = lift_data02,  
            class = "good") %>% 
    ggplot()
