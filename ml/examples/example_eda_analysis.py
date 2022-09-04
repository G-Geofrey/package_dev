
import pandas as pd

path='/Users/geofrey.wanyama/Library/CloudStorage/OneDrive-MoneseLtd/GW/projects/categorization/Data/salary2021Data.csv'

data_df = pd.read_csv(path)


from ml_processor import eda_data_quality

eda_data_quality(data_df)

from ml_processor import binary_eda_plot

plotColumns = {
    'target': 'target',
    'discrete': ['hourofday', 'dayofweek', 'dayofmonth', ],
    'numeric': ['trxnamount', 'sameotherpartywithinmonth', 'sameotherpartywithinweek', 'sameotherpartyall', 'paidsameweekquarter']
}

logColumns = ['trxnamount']

eda = binary_eda_plot(data_df, plotColumns, logColumns)

eda.get_plots()
