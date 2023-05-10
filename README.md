# Machine-Learning-Predicting-Goals
A project for my Data Science for Economists that explored if machine learning could predict if a goal is scored based on a set of variables from the hockeyR package.

**Project Description**

This project takes play by play data scraped from the NHL website in the [hockeyR package](https://github.com/danmorse314/hockeyR) created by Dan Morse. The data is filtered down to only the important variables for the regression.


**Data Collection and Cleaning**

1. Install hockeyR package 
2. Load in play by play data from one season 
3. Filter down play by play data to include SHOTS, MISSED_SHOT, and GOAL
  - This is all shots in a game
  - Blocked shots excluded as they do not result in goals
4. Create factors for special teams (SH and PP) and if a play is a goal
5. Remove unnecessary columns of data from the data frame/set

**Analysis Set Up**
1. Install and download machine learning packages that will run regression
  - rpart, e1071, kknn, and nnet
2. Set up respective algorithm and engine
3. Set up tuning parameters and cross validation per each model
4. Set up workflow that will test data based on given model
5. Print result from machine learning model
6. Store accuracy in table for later use
7. Repeat for other ML models
8. Print all accuracies in one table for output
9. Compare accuracies and penalty paramaters 

