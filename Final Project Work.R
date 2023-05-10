library(hockeyR)
library(tidyverse)
library(sportyR)
library(modelsummary)
library(tidymodels)
library(magrittr)
library(rpart)
library(e1071)
library(kknn)
library(nnet)

set.seed(1234)
# Load the play-by-play data for 2021-2022 season
pbp2022 <- load_pbp("2021-22")

# Filter the data to include only unblocked shots
shot_chart <- pbp2022 %>%
  filter(event_type %in% c("SHOT","MISSED_SHOT","GOAL"))


#Create factors for strength code and which plays are goals
shots$SH <- ifelse(shots$strength_code == "SH", 1, 0)
shots$PP <- ifelse(shots$strength_code == "PP", 1, 0)
shots$goal <- as.factor(ifelse(shots$event_type == "GOAL", 1, 0))

#Filter out unnecessary columns
shots %<>% select(-event_id, -secondary_type, -event_team, -event_team_type, -event_player_1_type, -event_player_3_type, -event_player_3_name, -penalty_severity, -penalty_minutes, -event_idx, -players_on, -players_off, -home_on_1, -home_on_2, -home_on_2, -home_on_3, -home_on_4, -home_on_5, -home_on_6, -home_on_7, -away_on_1, -away_on_2, -away_on_3, -away_on_4, -away_on_5, -away_on_6, -away_on_7, 
                  -game_id, -event_player_1_id, -event_player_1_link, -event_player_1_season_total, -event_player_2_id, -event_player_2_link, -event_player_2_season_total, -event_player_3_id, -event_player_3_link, -event_player_3_season_total, -event_player_4_id, -event_player_4_link, -date_time, -event_team_id, -event_team_link, -event_team_abbr, -home_final, -away_final, -event_goalie_link, -event,
                  -game_date, -game_start, -game_end, -game_length, -game_start, -detailed_state, -venue_id, -venue_name, -venue_link, -home_name, -home_id, -home_abbreviation, -home_division_name, -home_division_name_short, -home_conference_name, -home_id, -away_name, -away_abbreviation, -away_division_name, -away_division_name_short, -away_conference_name, -away_id,
                  -home_score, -away_score, -num_off, -num_on, -event_player_4_name, -event_player_4_type, -season, -season_type, -game_state, -description, -event_player_1_name, -event_goalie_name, -period_time, -period_time_remaining, -ordinal_num, -game_winning_goal, -strength_code, -event_goalie_type, -event_player_2_name, -event_player_2_type, -strength_state, -empty_net, -strength,
                  -x, -y, -x_fixed, -y_fixed, -home_skaters, -away_skaters, -home_goalie, -away_goalie, -event_goalie_id, -extra_attacker, -period_type, -event_type, -period_seconds_remaining, game_seconds_remaining, -xg, -game_seconds_remaining, -game_seconds)

shots <- drop_na(shots)

#Split data
shots_split <- initial_split(shots, prop = 0.8)
shots_train <- training(shots_split)
shots_test <- testing(shots_split)

## ML

####### 
# Logit 
tune_logit_spec <- logistic_reg(
  penalty = tune(), # tuning parameter
  mixture = 1       # 1 = lasso, 0 = ridge
) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")

# Lambda 
lambda_grid <- grid_regular(penalty(), levels = 50)

# 5-fold cross-validation
rec_folds_shots <- vfold_cv(shots_train, v = 5)

rec_wf_shots <- workflow() %>%
  add_model(tune_logit_spec) %>%
  add_formula(goal ~ shot_distance + shot_angle + period + period_seconds + SH + PP)

rec_res_shots <- rec_wf_shots %>%
  tune_grid(
    resamples = rec_folds_shots,
    grid = lambda_grid
  )

top_acc  <- show_best(rec_res_shots, metric = "accuracy")
best_acc <- select_best(rec_res_shots, metric = "accuracy")
final_logit_shots <- finalize_workflow(rec_wf_shots,
                                       best_acc
)

logit_test <- last_fit(final_logit_shots,shots_split) %>%
  collect_metrics()

logit_test %>% print(n = 1)
top_acc %>% print(n = 1)

# Save answer for later using tibble
logit_ans <- top_acc %>% slice(1)
logit_ans %<>% left_join(logit_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "logit") %>% select(-starts_with(".config"))

#######
# Tree 
tune_tree_spec <- decision_tree(
  min_n = tune(), # tuning parameter
  tree_depth = tune(), # tuning parameter
  cost_complexity = tune(), # tuning parameter
) %>% 
  set_engine("rpart") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (complexity, depth, etc.)
tree_parm_df1 <- tibble(cost_complexity = seq(.001,.2,by=.05))
tree_parm_df2 <- tibble(min_n = seq(10,100,by=10))
tree_parm_df3 <- tibble(tree_depth = seq(5,20,by=5))
tree_parm_df  <- full_join(tree_parm_df1,tree_parm_df2,by=character()) %>% full_join(.,tree_parm_df3,by=character())

# 5 fold CV
tree_ctrl <- vfold_cv(shots_train, v = 5)

rec_wf_tree_shots <- workflow() %>%
  add_model(tune_tree_spec) %>%
  add_formula(goal ~ shot_distance + shot_angle + period + period_seconds + SH + PP)

rec_res_tree <- rec_wf_tree_shots %>%
  tune_grid(
    resamples = tree_ctrl,
    grid = tree_parm_df
  )

top_acc_tree  <- show_best(rec_res_tree, metric = "accuracy")
best_acc_tree <- select_best(rec_res_tree, metric = "accuracy")
final_tree <- finalize_workflow(rec_wf_tree_shots,
                                best_acc_tree
)

print('*********** TREE MODEL **************')
tree_test <- last_fit(final_tree,shots_split) %>%
  collect_metrics()

tree_test %>% print(n = 1)
top_acc_tree %>% print(n = 1)

# Save answer for later using tibble
tree_ans <- top_acc_tree %>% slice(1)
tree_ans %<>% left_join(tree_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "Tree Model") %>% select(-starts_with(".config"))

#######
# Neural Net
tune_nnet_spec <- mlp(
  hidden_units = tune(), # tuning parameter
  penalty = tune()
) %>% 
  set_engine("nnet") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (number of neighbors)
nnet_parm_df1 <- tibble(hidden_units = seq(1,10))
lambda_grid   <- grid_regular(penalty(), levels = 10)
nnet_parm_df  <- full_join(nnet_parm_df1,lambda_grid,by=character())

# 5 Fold CV
nnet_ctrl <- vfold_cv(shots_train, v = 5)

rec_wf_nnet <- workflow() %>%
  add_model(tune_nnet_spec) %>%
  add_formula(goal ~ shot_distance + shot_angle + period + period_seconds + SH + PP)
    
rec_res_nnet <- rec_wf_nnet %>%
  tune_grid(
    resamples = nnet_ctrl,
    grid = nnet_parm_df
  )

top_acc_nnet  <- show_best(rec_res_nnet, metric = "accuracy")
best_acc_nnet <- select_best(rec_res_nnet, metric = "accuracy")
final_nnet <- finalize_workflow(rec_wf_nnet,
                                best_acc_nnet
)

print('*********** NEURAL NET **************')
nnet_test <- last_fit(final_nnet,shots_split) %>%
  collect_metrics()

nnet_test %>% print(n = 1)
top_acc_nnet %>% print(n = 1)

# Save answer for later using tibble
nnet_ans <- top_acc_nnet %>% slice(1)
nnet_ans %<>% left_join(nnet_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "Neural Net") %>% select(-starts_with(".config"))

#######
# KNN
print('Starting KNN')
# set up the task and the engine
tune_knn_spec <- nearest_neighbor(
  neighbors = tune() # tuning parameter
) %>% 
  set_engine("kknn") %>%
  set_mode("classification")

# define a set over which to try different values of the regularization parameter (number of neighbors)
knn_parm_df <- tibble(neighbors = seq(1,30))

# 5 fold CV
knn_ctrl <- vfold_cv(shots_train, v = 3)

rec_wf_knn <- workflow() %>%
  add_model(tune_knn_spec) %>%
  add_formula(goal ~ shot_distance + shot_angle + period + period_seconds + SH + PP)

rec_res_knn <- rec_wf_knn %>%
  tune_grid(
    resamples = knn_ctrl,
    grid = knn_parm_df
  )

top_acc_knn  <- show_best(rec_res_knn, metric = "accuracy")
best_acc_knn <- select_best(rec_res_knn, metric = "accuracy")
final_knn <- finalize_workflow(rec_wf_knn,
                               best_acc_knn
)

print('*********** KNN **************')
knn_test <- last_fit(final_knn,shots_split) %>%
  collect_metrics()

knn_test %>% print(n = 1)
top_acc_knn %>% print(n = 1)

# Save answer for later using tibble
knn_ans <- top_acc_knn %>% slice(1)
knn_ans %<>% left_join(knn_test %>% slice(1),by=c(".metric",".estimator")) %>%
  mutate(alg = "KNN") %>% select(-starts_with(".config"))

# Display all answers
all_ans <- bind_rows(logit_ans,tree_ans,nnet_ans,knn_ans)
datasummary_df(all_ans %>% select(-.metric,-.estimator,-mean,-n,-std_err),output="markdown") %>% print


