# MarketPrediction
 
 Data_Pipeline.py:
    1. Create all of the directory file paths for the files to be saved later
    2. Takes in all of the data in the form of a CSV. Cleans the data by making sure that all columns necessary are there and drops rows that are missing any values. Data comes in the form of information on specific publically traded stocks as well as macroeconimic factors 
    3. Computes trading days, allowing us to only look at days that we are able to make trades
    4. Creates leak-proof features that will allow us to train the ML model without using data from time t + n to make decisions at time t. This ensures that it is actually learning and not just looking backwards. 
    5. Create the data values that the model will use to create the regression that will predict the prices for future trading days. By having the model make predictions about t + n at time t and then checking the outcomes, we have created a supervised learning enviornment. 

Train_baseline.py:
    ---- Meant to be a basic sanity check that the data pipeline is working---- 

    1. Load the features and labels that were created for each stock
    2. Creates a simple regression model with accuracy number to ensure that the data pipeline is working.
    3. Trains the model on the first 80% of the data and then tests on the last 20% to produce a sanity check score. 


backtest_walk_forward.py:
    1. Builds a training set from set number of days 
    2. Uses the training days to fit a regression model that will then make an estimate of whether the stock price will increase tomorrow based on todays features
    3. If the probability if the stock going up is above a certain percentage, go long on the stock, else do nothing. Incresing the probability threshold would cause lower turnover, but could miss out on winning trades
    4. Create a metric output that makes it easy to look over model results for easier tuning. 

run_pipeline.sh
    Bash script that can be run to automotically make all necessary function calls to streamline the running process. 

ablation_runner.py
    Runs different combonations of training parameters to find optimal training paramters for sharpe rating 
