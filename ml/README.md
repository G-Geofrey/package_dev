<!-- <p style=" border-top: 5px solid #21B6A8; 
        ">
    ml_processor package
</p> -->
<h1 style="text-align: left; background-color:  #21B6A8; font-size: 32px; font-weight:bold; padding: 10px; 
          font-family: Tahoma, sans-serif;">
    ml_processor package
</h1>

## Configuration


```python
from ml_processor.configuration import config
```

<div class="module" >
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8; padding: 10px;">
            <span style="font-size: 16px; font-weight:bold"><b>config</b></span>
        </p>
        <p style="margin-left:25px">
            Perform basic configurations and logging
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li>None</li>
            </ul>
        </p>
    </div>
</div>

<div class="module" >
    <!-- add path -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>add_path</b></span> (lib_path=None)
            </span>
        </p>
        <p style="margin-left:25px">
            Add path to current home path
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>lib_path</b> (<i>string</i>) - Path to add to home path</li>
            </ul>
        </p>
    </div>
</div>



<!-- get_credentials -->
<div class="module-details", style="margin-left:25px;">
    <p style="">
        <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
            <span style="font-size: 16px; font-weight:bold"><b>get_credentials</b></span> (env_path=None)
        </span>
    </p>
    <p style="margin-left:25px">
        Get credentials stored in dot file
    </p>
    <p >
        <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
            Parameters
        </span>
    </p>
    <p style="margin-left:25px">
        <ul style="margin-left:50px">
            <li><b>env_path</b> (string) - Path to dot file with credentials</li>
        </ul>
    </p>
    <p>
        <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
            Returns 
        </span>
    </p>
    <p style="margin-left:50px; ">
        <b>dict</b> - Stored credentials 
    </p>
    <p >
        <span style="margin-left:25px; padding: 5px; font-weight: bold">
            Example
        </span>
    </p>
</div>



```python
env_path = './.env'
config.get_credentials(env_path)
```




    OrderedDict([('username', 'email.example.com'), ('password', 'myPassword')])



<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>get_logger</b></span> (file=None)
            </span>
        </p>
        <p style="margin-left:25px">
            Create logging.getLogger() object
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>file</b> (<i>string (default=None)</i>) - File name to store log e.g modelLogs.log. If no file name is provided, logging is on done to the console while if the file name is provide, logging is done both to the console and file</li>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>object</b> - logging.getLogger 
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>






```python
logger = config.get_logger()
logger.info('Looking to console only. Provide file name to create log file')
```

    2022-09-30 17:20:15,143:INFO:Looking to console only. Provide file name to create log file


## snowflake_processor


```python
from ml_processor.etl_processor import snowflake_processor
```

<div class="module" >
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8; padding: 10px;">
            <span style="font-size: 16px; font-weight:bold"><b>snowflake_processor</b></span>(username=None, password=None, account=None, warehouse=None, database=None)
        </p>
        <p style="margin-left:25px">
            Performing ETL tasks such as connecting to snowflake and retrieving data from snowflake.
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>username</b> (<i>string (default=None)</i>) - Username for connecting to snowflake.</li><br>
                <li><b>password</b> (<i>string (default=None)</i>) - Password for connecting to snowflake.</li><br>
                <li><b>account</b> (<i>string (default=None) </i>) - Snowflake account.</li><br>
                <li><b>warehouse</b> (<i>string (default=None)</i>) - Warehouse name.</li><br>
                <li><b>database</b> (<i>string (default=None)</i>) - Database name.</li><br>
            </ul>
        </p>
    </div>
</div>

<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>connect</b></span> ( )
            </span>
        </p>
        <p style="margin-left:25px">
            Create connection to snowflake.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>None</b></li> 
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>object</b> - connection to snowflake.
        </p>
    </div>
</div>


<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>pandas_from_sql</b></span> (sql, conn=None, chunksize=None)
            </span>
        </p>
        <p style="margin-left:25px">
            Extracting data from snowflake into pandas dataframe.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>sql</b> (<i>string</i>) - Dataset to balance.</li><br>
                <li><b>conn</b> (<i>object (default=None) </i>) - Connection engine to snowflake.</li><br>
                <li><b>chunksize</b> (<i>int (default=None)</i>) - Number of rows to extract from snowflake per iteration if extracting in chunks. If chunksize is not provided, all rows are extracted</li><br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>pandas.DataFrame</b> - Data extracted fro snowflake.
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
conn = snowflake_processor()
df_extract = conn.pandas_from_sql(sql_test)
```

## eda_data_quality


```python
from ml_processor.eda_analysis import eda_data_quality
```

<div class="module" >
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8; padding: 10px;">
            <span style="font-size: 16px; font-weight:bold"><b>eda_data_quality</b></span>(data)
        </p>
        <p style="margin-left:25px">
            Performing data quality checks on data set.
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>data</b> (<i>pandas.DataFrame</i>) - Data set to check.</li> <br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>


```python
# read in some data
import pandas as pd
df = pd.read_csv('./data/application_train.csv')

eda_data_quality(df).head()
```

    2022-09-30 20:17:15,974:INFO:rule_1 : More than 50% of the data missing
    2022-09-30 20:17:15,975:INFO:rule_2 : Missing some data
    2022-09-30 20:17:15,975:INFO:rule_3 : 75% of the data is the same and equal to the minimum
    2022-09-30 20:17:15,976:INFO:rule_4 : 50% of the data is the same and equal to the minimum
    2022-09-30 20:17:15,976:INFO:rule_5 : Has negative values
    2022-09-30 20:17:15,976:INFO:rule_6 : Possible wrong data type





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>unique</th>
      <th>missing</th>
      <th>pct.missing</th>
      <th>mean</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>rule_1</th>
      <th>rule_2</th>
      <th>rule_3</th>
      <th>rule_4</th>
      <th>rule_5</th>
      <th>rule_6</th>
      <th>quality_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>reg_city_not_live_city</th>
      <td>int64</td>
      <td>2</td>
      <td>0</td>
      <td>0.0%</td>
      <td>0.0372</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6429</td>
    </tr>
    <tr>
      <th>amt_req_credit_bureau_qrt</th>
      <td>float64</td>
      <td>8</td>
      <td>0</td>
      <td>0.0%</td>
      <td>0.2581</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>8.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6429</td>
    </tr>
    <tr>
      <th>flag_document_10</th>
      <td>int64</td>
      <td>2</td>
      <td>0</td>
      <td>0.0%</td>
      <td>0.0001</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6429</td>
    </tr>
    <tr>
      <th>flag_mobil</th>
      <td>int64</td>
      <td>1</td>
      <td>0</td>
      <td>0.0%</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6429</td>
    </tr>
    <tr>
      <th>flag_document_4</th>
      <td>int64</td>
      <td>1</td>
      <td>0</td>
      <td>0.0%</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6429</td>
    </tr>
  </tbody>
</table>
</div>



## binary_eda_plot


```python
from ml_processor.eda_analysis import binary_eda_plot
```

<div class="module" >
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8; padding: 10px;">
            <span style="font-size: 16px; font-weight:bold"><b>binary_eda_plot</b></span>(data, 
                 target='target', 
                 plot_columns=None, 
                 log_columns=[None], 
                 exclude_cols=[None], 
                 columns=6, 
                 target_palette={1:'red', 0:'deepskyblue'},
                 bin_numeric=True)
        </p>
        <p style="margin-left:25px">
            Visualizing data for explatory analysis.
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>data</b> (<i>pandas.DataFrame</i>) - Data set for explatory analysis.</li><br>
                <li><b>plot_columns</b> (<i>dict (default=None)</i>) - Columns to visualize.</li><br>
                <li><b>log_columns</b> (<i>list (default=[None])</i>) - Columns to use log scale.</li><br>
                <li><b>exclude_cols</b> (<i>list (default=[None])</i>) - Columns to not to plot.</li><br>
                <li><b>columns</b> (<i>int (default=6)</i>) - Number of columns in the matplotlib subplots.</li><br>
                <li><b>target_palette</b> (<i>dict (default = {1:'red', 0:'deepskyblue'})</i>) - Palette for the labels.</li> <br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>matplotlib plot</b> - Figure of plotted columns
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>


```python
eda_plot = binary_eda_plot(df[check_columns])
eda_plot.get_plots()
```


    
![png](output_22_0.png)
    


## data_prep


```python
from ml_processor.data_prep import data_prep
```

<div class="module" >
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8; padding: 10px;">
            <span style="font-size: 16px; font-weight:bold"><b>data_prep</b></span>(data, features, target='target', categories=None)
        </p>
        <p style="margin-left:25px">
            Data preparation to transform data using one hot encoding or woe transformation.
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>data</b> (<i>pandas.DataFrame</i>) - Data set to transform</li> <br>
                <li><b>features</b> (<i>list or array-like</i>) - Names of columns (variables) to transform using either one hot encoding or woe transformation</li><br>
                <li><b>target</b> (<i>string (default='target')</i>) - Name of the column with binary labels</li><br>
                <li><b>categories</b> (<i>list or array-like (default='target')</i>) - Names of categorical variables in the dataset. </li><br>
            </ul>
        </p>
    </div>
</div>

<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>encoder</b></span> ( )
            </span>
        </p>
        <p style="margin-left:25px">
            Create encode for one hot encoding
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>drop</b> (<i>string (default='if_binary', possible values {'first', 'if_binary'} or an array-like of shape (n_features,))</i>) - Specifies a methodology to use to drop one of the categories per feature. </li>
                <li><b>verbose</b> (<i>boolean (default=False)</i>) - Log to console or not</li>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>self</b> - One Hot Encoder. The encoder is saved as <u><i>encoder</i></u> in a folder named <u><i>data_prep</i></u> in the current working directory. 
        </p>
    </div>
</div>





<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>oneHot_transform</b></span> (file=None)
            </span>
        </p>
        <p style="margin-left:25px">
            Transform data using one hot encoding
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>data</b> (<i>pandas.DataFrame (default=empty dataframe)</i>) - Data to transfrom using one hot encoding</li>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>pandas.DataFrame</b> - Transformed dataset 
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
# define the variables
target = 'target' 

all_features = ['amt_income_total', 'name_contract_type','code_gender']

categories = ['name_contract_type','code_gender']

# transform just a few columns - include full data set for complete transformation
check_columns = [target] + all_features 

# initiate data transformation
init_data = data_prep1(data=df[check_columns], features=features, categories=categories)

# get transformed  data
df_encode = init_data.oneHot_transform()
df_encode.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>amt_income_total</th>
      <th>name_contract_type</th>
      <th>code_gender</th>
      <th>name_contract_type_Revolving loans</th>
      <th>code_gender_M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>103500.0000</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>202500.0000</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>202500.0000</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>162000.0000</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>225000.0000</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                enc
            </span> - Get One Hot Encoder
        </p>
    </div>
</div>



```python
init_data.enc
```




    OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False)



<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>woe_bins</b></span> (min_prebin_size = 0.1, 
                 selection_criteria = {"iv": {"min": 0.01, "max": 0.7, "strategy": "highest", "top": 50},
                                       "quality_score": {"min": 0.01}},
                 binning_fit_params = None,
                 verbose=False)
            </span>
        </p>
        <p style="margin-left:25px">
            Generate binning process for woe transformation
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>min_prebin_size</b> (<i>float (default=0.1)</i>) - The fraction of minimum number of records for each bin</li> <br>
                <li><b>selection_criteria</b> (<i>dict or None (default={"iv": {"min": 0.01, "max": 0.7, "strategy": "highest", "top": 50},
                                       "quality_score": {"min": 0.01}})</i>) - Variable selection criteria</li><br>
                <li><b>binning_fit_params</b> (<i>dict or None (default=None)</i>) - Dictionary with optimal binning transform options for specific variables. Example ``{"variable_1": {"metric": "event_rate"}}``.
            </li><br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>Self</b> - Fitted binning process.
        </p>
    </div>
</div>


<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>woe_bin_table</b></span> ( )
            </span>
        </p>
        <p style="margin-left:25px">
            Generate summary results for the binning process
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>None</b></li> 
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>pandas.DataFrame</b> - Summary results on the binning process.
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
target = 'target'
features = [col for col in df.columns if col != 'target']
categories = [col for col in df.columns if str(df[col].dtype)=='object' and col != 'target']

# initiate data transformation
init_data = data_prep1(data=df, features=features, categories=categories)

# get transformed  data
bin_table = init_data.woe_bin_table()
bin_table.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>dtype</th>
      <th>status</th>
      <th>selected</th>
      <th>n_bins</th>
      <th>iv</th>
      <th>js</th>
      <th>gini</th>
      <th>quality_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ext_source_3</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>True</td>
      <td>6</td>
      <td>0.3172</td>
      <td>0.0386</td>
      <td>0.3078</td>
      <td>0.9270</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ext_source_2</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>True</td>
      <td>8</td>
      <td>0.3078</td>
      <td>0.0374</td>
      <td>0.3060</td>
      <td>0.9191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ext_source_1</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>True</td>
      <td>3</td>
      <td>0.1271</td>
      <td>0.0156</td>
      <td>0.1695</td>
      <td>0.3876</td>
    </tr>
    <tr>
      <th>3</th>
      <td>days_employed</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>True</td>
      <td>7</td>
      <td>0.1109</td>
      <td>0.0138</td>
      <td>0.1846</td>
      <td>0.3997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>days_birth</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>True</td>
      <td>7</td>
      <td>0.0852</td>
      <td>0.0106</td>
      <td>0.1636</td>
      <td>0.3296</td>
    </tr>
  </tbody>
</table>
</div>



<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>get_var_bins</b></span> (var)
            </span>
        </p>
        <p style="margin-left:25px">
            Generate binning details for the variable.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>var</b> (<i>string</i>) - Name of variable for which to show binning tables</li> <br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>matplotlib plot</b> - Bar chart showing bins for the var
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
init_data.get_var_bins('ext_source_3')
```


    
![png](output_35_0.png)
    


<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>woe_transform</b></span> (data=pd.DataFrame(), verbose=False)
            </span>
        </p>
        <p style="margin-left:25px">
            Transform data using Weight of Evidence (WOE) weights.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>data</b> (<i>pandas.DataFrame (default=empty dataframe)</i>) - Dataset to tranform</li> <br>
                <li><b>verbose</b> (<i>boolean (default=False)</i>) - Log to console or not.</li> <br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>pandas.DataFrame</b> - Transformed data.
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
df_woe = init_data.woe_transform()
df_woe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code_gender</th>
      <th>amt_credit</th>
      <th>amt_annuity</th>
      <th>amt_goods_price</th>
      <th>name_income_type</th>
      <th>name_education_type</th>
      <th>region_population_relative</th>
      <th>days_birth</th>
      <th>days_employed</th>
      <th>days_registration</th>
      <th>days_id_publish</th>
      <th>flag_emp_phone</th>
      <th>occupation_type</th>
      <th>region_rating_client</th>
      <th>region_rating_client_w_city</th>
      <th>reg_city_not_work_city</th>
      <th>organization_type</th>
      <th>ext_source_1</th>
      <th>ext_source_2</th>
      <th>ext_source_3</th>
      <th>apartments_avg</th>
      <th>basementarea_avg</th>
      <th>years_beginexpluatation_avg</th>
      <th>elevators_avg</th>
      <th>entrances_avg</th>
      <th>floorsmax_avg</th>
      <th>livingarea_avg</th>
      <th>nonlivingarea_avg</th>
      <th>apartments_mode</th>
      <th>basementarea_mode</th>
      <th>years_beginexpluatation_mode</th>
      <th>elevators_mode</th>
      <th>entrances_mode</th>
      <th>floorsmax_mode</th>
      <th>livingarea_mode</th>
      <th>nonlivingarea_mode</th>
      <th>apartments_medi</th>
      <th>basementarea_medi</th>
      <th>years_beginexpluatation_medi</th>
      <th>elevators_medi</th>
      <th>entrances_medi</th>
      <th>floorsmax_medi</th>
      <th>livingarea_medi</th>
      <th>nonlivingarea_medi</th>
      <th>housetype_mode</th>
      <th>totalarea_mode</th>
      <th>wallsmaterial_mode</th>
      <th>emergencystate_mode</th>
      <th>days_last_phone_change</th>
      <th>flag_document_3</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.2509</td>
      <td>-0.2043</td>
      <td>-0.0836</td>
      <td>-0.4453</td>
      <td>-0.1892</td>
      <td>-0.1130</td>
      <td>-0.1838</td>
      <td>-0.3721</td>
      <td>-0.3785</td>
      <td>-0.0784</td>
      <td>-0.0886</td>
      <td>-0.0766</td>
      <td>-0.3315</td>
      <td>0.0250</td>
      <td>0.0211</td>
      <td>0.1071</td>
      <td>-0.1567</td>
      <td>-0.5758</td>
      <td>-0.4351</td>
      <td>-0.8393</td>
      <td>-0.0624</td>
      <td>0.0582</td>
      <td>0.0036</td>
      <td>0.0549</td>
      <td>0.0127</td>
      <td>-0.0901</td>
      <td>-0.0734</td>
      <td>0.1155</td>
      <td>-0.0761</td>
      <td>0.0773</td>
      <td>0.0232</td>
      <td>0.0653</td>
      <td>0.0456</td>
      <td>-0.0865</td>
      <td>-0.0862</td>
      <td>0.1273</td>
      <td>-0.0597</td>
      <td>0.0623</td>
      <td>0.0000</td>
      <td>0.0573</td>
      <td>0.0172</td>
      <td>-0.0896</td>
      <td>-0.0725</td>
      <td>0.1223</td>
      <td>0.1562</td>
      <td>-0.0993</td>
      <td>0.0752</td>
      <td>0.1538</td>
      <td>0.1817</td>
      <td>-0.0998</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1543</td>
      <td>0.4644</td>
      <td>-0.0387</td>
      <td>0.5296</td>
      <td>0.4136</td>
      <td>0.4411</td>
      <td>-0.0488</td>
      <td>0.0575</td>
      <td>-0.2565</td>
      <td>-0.0784</td>
      <td>-0.2590</td>
      <td>-0.0766</td>
      <td>0.2407</td>
      <td>0.5504</td>
      <td>0.5460</td>
      <td>0.1071</td>
      <td>0.4320</td>
      <td>-0.5758</td>
      <td>0.3964</td>
      <td>0.0000</td>
      <td>0.2160</td>
      <td>0.1460</td>
      <td>0.1862</td>
      <td>0.3648</td>
      <td>0.0127</td>
      <td>0.3657</td>
      <td>0.1066</td>
      <td>0.1824</td>
      <td>0.2055</td>
      <td>0.1671</td>
      <td>0.1824</td>
      <td>0.3660</td>
      <td>0.0456</td>
      <td>0.3675</td>
      <td>0.1243</td>
      <td>0.1273</td>
      <td>0.2153</td>
      <td>0.1321</td>
      <td>0.1843</td>
      <td>0.3669</td>
      <td>0.0172</td>
      <td>0.3675</td>
      <td>0.1157</td>
      <td>0.1786</td>
      <td>0.1562</td>
      <td>0.1487</td>
      <td>0.0752</td>
      <td>0.1538</td>
      <td>-0.0191</td>
      <td>-0.0998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.2509</td>
      <td>0.1978</td>
      <td>0.1802</td>
      <td>0.1067</td>
      <td>-0.1892</td>
      <td>-0.1130</td>
      <td>-0.0488</td>
      <td>0.1631</td>
      <td>-0.3349</td>
      <td>-0.0744</td>
      <td>-0.0785</td>
      <td>-0.0766</td>
      <td>-0.3315</td>
      <td>0.0250</td>
      <td>0.0211</td>
      <td>0.1071</td>
      <td>0.1389</td>
      <td>0.0000</td>
      <td>0.2066</td>
      <td>0.9492</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0191</td>
      <td>0.2871</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1543</td>
      <td>-0.2043</td>
      <td>-0.2491</td>
      <td>-0.4453</td>
      <td>-0.1892</td>
      <td>-0.1130</td>
      <td>-0.0488</td>
      <td>0.1631</td>
      <td>0.1858</td>
      <td>0.3725</td>
      <td>-0.0886</td>
      <td>-0.0766</td>
      <td>-0.3315</td>
      <td>0.0250</td>
      <td>0.0211</td>
      <td>0.1071</td>
      <td>-0.1567</td>
      <td>0.0000</td>
      <td>0.3964</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.1508</td>
      <td>-0.0998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.2509</td>
      <td>-0.2043</td>
      <td>-0.0836</td>
      <td>0.0475</td>
      <td>-0.1892</td>
      <td>-0.1130</td>
      <td>0.0390</td>
      <td>0.3947</td>
      <td>0.1858</td>
      <td>-0.0744</td>
      <td>0.0288</td>
      <td>-0.0766</td>
      <td>0.2407</td>
      <td>0.0250</td>
      <td>0.0211</td>
      <td>-0.3014</td>
      <td>0.4320</td>
      <td>0.0000</td>
      <td>-0.4351</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0191</td>
      <td>0.2871</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>woe_features</b></span> (verbose=False)
            </span>
        </p>
        <p style="margin-left:25px">
            Generate variables selected using the selection criteria
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>verbose</b> (<i>boolean (default=False)</i>) - Log to console or not.</li> <br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>array</b> - variables selected.
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
woe_features = init_data.woe_features()
woe_features
```




    array(['code_gender', 'amt_credit', 'amt_annuity', 'amt_goods_price',
           'name_income_type', 'name_education_type',
           'region_population_relative', 'days_birth', 'days_employed',
           'days_registration', 'days_id_publish', 'flag_emp_phone',
           'occupation_type', 'region_rating_client',
           'region_rating_client_w_city', 'reg_city_not_work_city',
           'organization_type', 'ext_source_1', 'ext_source_2',
           'ext_source_3', 'apartments_avg', 'basementarea_avg',
           'years_beginexpluatation_avg', 'elevators_avg', 'entrances_avg',
           'floorsmax_avg', 'livingarea_avg', 'nonlivingarea_avg',
           'apartments_mode', 'basementarea_mode',
           'years_beginexpluatation_mode', 'elevators_mode', 'entrances_mode',
           'floorsmax_mode', 'livingarea_mode', 'nonlivingarea_mode',
           'apartments_medi', 'basementarea_medi',
           'years_beginexpluatation_medi', 'elevators_medi', 'entrances_medi',
           'floorsmax_medi', 'livingarea_medi', 'nonlivingarea_medi',
           'housetype_mode', 'totalarea_mode', 'wallsmaterial_mode',
           'emergencystate_mode', 'days_last_phone_change', 'flag_document_3'],
          dtype='<U28')



<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>balance_data</b></span> (data=pd.DataFrame())
            </span>
        </p>
        <p style="margin-left:25px">
            Balance data basing on each label size of the label variable.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>data</b> (<i>pandas.DataFrame (default=empty dataframe))</i>) - Dataset to balance.</li> <br>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>pandas.DataFrame</b> - Transformed data.
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
df_balanced = init_data.balance_data(df_woe)
```


    
![png](output_41_0.png)
    


<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                binning_process
            </span> - Get binning process
        </p>
    </div>
</div>



```python
init_data.binning_process
```




    BinningProcess(categorical_variables=['name_contract_type', 'code_gender',
                                          'flag_own_car', 'flag_own_realty',
                                          'name_type_suite', 'name_income_type',
                                          'name_education_type',
                                          'name_family_status', 'name_housing_type',
                                          'occupation_type',
                                          'weekday_appr_process_start',
                                          'organization_type', 'fondkapremont_mode',
                                          'housetype_mode', 'wallsmaterial_mode',
                                          'emergencystate_mo...
                                   'name_education_type', 'name_family_status',
                                   'name_housing_type',
                                   'region_population_relative', 'days_birth',
                                   'days_employed', 'days_registration',
                                   'days_id_publish', 'own_car_age', 'flag_mobil',
                                   'flag_emp_phone', 'flag_work_phone',
                                   'flag_cont_mobile', 'flag_phone', 'flag_email',
                                   'occupation_type', 'cnt_fam_members',
                                   'region_rating_client',
                                   'region_rating_client_w_city', ...])



## xgbmodel


```python
from ml_processor.model_training import xgbmodel
```

<div class="module" >
    <div class="module-details">
        <p style="background-color:  #D4F1F4; border-top: 5px solid #21B6A8; padding: 10px;">
            <span style="font-size: 16px; font-weight:bold"><b>xgbmodel</b></span>(df, features, target, params_prop=0.25, test_size=0.33, hyperparams=None, scoring='recall')
        </p>
        <p style="margin-left:25px">
            Performing machine learning tasks including hyperparameter tuning and xgb model fitting.
        </p>
        <p style="">
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; 
                         padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>df</b> (<i>pandas.Dataframe</i>) - Dataset with features and labels.</li><br>
                <li><b>features</b> (<i>list or array-like</i>) - Variable names (features) for fitting the model.</li><br>
                <li><b>target</b> (<i>string</i>) - Name of column with labels (dependent variable).</li><br>
                <li><b>params_prop</b> (<i>float (default=0.25)</i>) - Proportion of data set to use for hyperparameter tuning.</li><br>
                <li><b>test_size</b> (<i>float (default=0.33)</i>) - Proportion of data to use as the test set.</li><br>
                <li><b>hyperparams</b> (<i>dictionary (default=None) </i>) - Predefined hyperparameters and their values.Specified if hyperparameter tunning is not necessary.</li><br>
                <li><b>scoring</b> (<i>string</i>) - Performance metric to maximises.</li><br>
            </ul>
        </p>
    </div>
</div>

<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>model_results</b></span> ( )
            </span>
        </p>
        <p style="margin-left:25px">
            Fitting model and performing model diagnostics.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>None</b></li>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>object</b> - xgb model object.
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
# initiate model
xgb_woe = xgbmodel(df_balanced
                   ,features=woe_features
                   ,target=target
                   ,hyperparams=None
                   ,test_size=0.25
                   ,params_prop=0.1
                      )
```

    2022-09-30 22:52:30,512:INFO:Model job name: xgbmodel_20220930225230



```python
# Fit model and get model results
xgb_model = xgb_woe.model_results()
```

    2022-09-30 22:52:31,854:INFO:Splitting data into training and testing sets completed
    2022-09-30 22:52:31,854:INFO:Training data set:37237 rows
    2022-09-30 22:52:31,855:INFO:Testing data set:12413 rows
    2022-09-30 22:52:31,855:INFO:Hyper parameter tuning data set created
    2022-09-30 22:52:31,856:INFO:Hyper parameter tuning data set:4965 rows
    2022-09-30 22:52:31,862:INFO:Splitting hyperparameter tuning data into training and testing sets completed
    2022-09-30 22:52:31,863:INFO:Hyperparameter tuning training data set:3723 rows
    2022-09-30 22:52:31,864:INFO:Hyperparameter tuning testing data set:1242 rows
    2022-09-30 22:52:31,865:INFO:Trials initialized...
    100%|████████| 48/48 [00:30<00:00,  1.59trial/s, best loss: -0.6884176182707994]
    2022-09-30 22:53:02,001:INFO:Hyperparameter tuning completed
    2022-09-30 22:53:02,002:INFO:Runtime for Hyperparameter tuning : 0 seconds
    2022-09-30 22:53:02,003:INFO:Best parameters: {'colsample_bytree': 0.6000000000000001, 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 7, 'reg_alpha': 1, 'reg_lambda': 10}
    2022-09-30 22:53:02,004:INFO:Model fitting initialized...
    2022-09-30 22:53:02,004:INFO:Model fitting started...
    2022-09-30 22:53:05,231:INFO:Model fitting completed
    2022-09-30 22:53:05,232:INFO:Runtime for fitting the model : 19 seconds
    2022-09-30 22:53:05,236:INFO:Model saved: /Users/geofrey.wanyama/Desktop/libraries/ml/examples/model_artefacts/xgbmodel_20220930225230.sav
    2022-09-30 22:53:05,242:INFO:Dataframe with feature importance generated
    2022-09-30 22:53:05,271:INFO:Predicted labels generated (test)
    2022-09-30 22:53:05,295:INFO:Predicted probabilities generated (test)
    2022-09-30 22:53:05,300:INFO:Confusion matrix generated (test)
    2022-09-30 22:53:05,305:INFO:AUC (test): 74%
    2022-09-30 22:53:05,309:INFO:Precision (test): 68%
    2022-09-30 22:53:05,310:INFO:Recall (test): 68%
    2022-09-30 22:53:05,311:INFO:F_score (test): 68%
    2022-09-30 22:53:05,313:INFO:Precision and Recall values for the precision recall curve created
    2022-09-30 22:53:05,316:INFO:True positive and negativevalues for the ROC curve created
    2022-09-30 22:53:05,375:INFO:Recall and precision calculation for different thresholds (test) completed


<div class="module" >
    <!-- logger -->
    <div class="module-details", style="margin-left:25px;">
        <p style="">
            <span style="background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                <span style="font-size: 16px; font-weight:bold"><b>make_plots</b></span> ( )
            </span>
        </p>
        <p style="margin-left:25px">
            Fitting model and performing model diagnostics.
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Parameters
            </span>
        </p>
        <p style="margin-left:25px">
            <ul style="margin-left:50px">
                <li><b>None</b></li>
            </ul>
        </p>
        <p >
            <span style="margin-left:25px; background-color:#ECECEC; border-left: 2.5px solid #D4D4D4; padding: 5px">
                Returns 
            </span>
        </p>
        <p style="margin-left:50px; ">
            <b>Matplotlib plot</b> 
        </p>
        <p >
            <span style="margin-left:25px; padding: 5px; font-weight: bold">
                Example
            </span>
        </p>
    </div>
</div>



```python
# Model evalaution
xgb_woe.make_plots()
```


    
![png](output_51_0.png)
    



```python

```
