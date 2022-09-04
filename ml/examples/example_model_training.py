
import pandas as pd

path='/Users/geofrey.wanyama/Library/CloudStorage/OneDrive-MoneseLtd/GW/projects/categorization/Data/salary2021Data.csv'

data_df = pd.read_csv(path)

minority_size = data_df.target.value_counts()[1]

data_bal = data_df.groupby('target').sample(minority_size)

# shuffled the data - just want the data randomized
data_bal = data_bal.sample(frac=1)

features = data_bal.drop(['cusreference','trxnid','trxnsettlementdate', 'target'], axis=1).columns.to_list()

target = 'target'

from ml_processor import xgb_training

params = {'colsample_bytree': 0.6000000000000001, 
          'gamma': 0.30000000000000004, 
          'learning_rate': 0.1, 
          'max_depth': 11, 
          'reg_alpha': 0.01, 
          'reg_lambda': 1
         }

results = xgb_training(data_bal.groupby('target').sample(5000)
                       , features=features, target=target, params_prop=0.25, hyperparams=params)

results.model_results()

results.make_plots()