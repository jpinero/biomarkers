rm(list = ls())
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(tidytext)
set.seed(1234)

corpus <- data.table::fread("data/corpus_sample.csv")
corpus$p1 <- unlist( str_locate(pattern = corpus$text,corpus$measurement))[,1]
corpus$p2 <- unlist( str_locate(pattern = corpus$text,corpus$measurement))[,2]
corpus$sentence <- paste0(substr(corpus$measurement, 0, corpus$p1-1),"biomarker", substr(corpus$measurement, corpus$p2+1, nchar(corpus$measurement)))

corpus <- corpus %>%
  mutate(label = factor( label, levels = c("protein", "genetic",  "epigenetic",     "gene expression",
                                           "cell surface", "phosphorylation")))


text_split <- initial_split(corpus, strata = label)

training_set <- training(text_split)
test_set <- testing(text_split)

dim(training_set)
dim(test_set )

text_recipe <-
  recipe(label ~ sentence,  
         data = training_set) %>%
  step_tokenize(sentence) %>%
  step_stopwords(sentence) %>%
  step_tokenfilter(sentence,
                   max_tokens = tune(),
                   min_times = 10) %>%
  step_tfidf(sentence)


param_grid <- grid_regular(
  max_tokens(range = c(
    50,1500)),
  mtry(range = c(
    1,100)),
  trees(range = c(500, 1500)),
  min_n(range = c(2, 8)),
  levels=5
)


dat_folds <- vfold_cv(data = training_set, strata = label)
dat_folds

set.seed(123)

library(ranger)
rf_spec <- 
  rand_forest( mtry = tune(),
               trees = tune(),
               min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

rf_wflow <-
  workflow() %>%
  add_recipe(text_recipe) %>% 
  add_model(rf_spec) 
 
tune_rf <- tune_grid(
  rf_wflow,
  dat_folds,
  grid = param_grid,
  metrics = metric_set(
    recall, precision, f_meas, 
    accuracy, kap,
    roc_auc, sens, spec),
  control = control_resamples(save_pred = TRUE)
)


tune_rf %>%  collect_metrics(summarize = TRUE)
rf_predictions <- collect_predictions(tune_rf)


rf_predictions %>%
  filter(id == "Fold01") %>%
  conf_mat(label, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))


rf_metrics <- 
  tune_rf %>% 
  collect_metrics(summarise = TRUE) %>%
  mutate(model = "Random Forest")



show_best(tune_rf, metric = "accuracy") %>%
  knitr::kable()


tune_rf %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, trees, mtry) %>%
  pivot_longer(trees:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

tune_rf %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

best_accuracy <- select_best(tune_rf, "accuracy")

rf_wf_final <- finalize_workflow(
  rf_wflow,
  best_accuracy
)


last_fit_rf <- last_fit(rf_wf_final, 
                        split = text_split,
                        metrics = metric_set(
                          recall, precision, f_meas, 
                          accuracy, kap,
                          roc_auc, sens, spec)
)



final_res_metrics <- collect_metrics(last_fit_rf)
final_res_predictions <- collect_predictions(last_fit_rf)

final_res_metrics %>%
  knitr::kable()

final_res_predictions %>%
  conf_mat(truth = label, estimate = .pred_class) %>%
  autoplot(type = "heatmap")



library(vip)

last_fit_rf %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip(num_features = 20) + theme_bw()

last_fit_rf %>%
  collect_predictions() %>% 
  conf_mat(label, .pred_class) %>% 
  autoplot(type = "heatmap")


final_model <- fit(rf_wf_final, corpus)

preds <- final_model %>%
  predict(new_data = test_set)

test_set <- bind_cols( test_set, preds) %>%
  mutate(correct = (.pred_class == label))

test_set %>%
  summarise(sum(correct) / n())




 
 

