
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from optbinning import BinningProcess
import os
import pickle
import warnings
import time

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

from ml_processor.configuration import config

# <<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class data_prep:

    """
    Data preparation to transform data using one hot encoding or woe transformation.

    Parameters
    ----------
    data : pandas.DataFrame 
        Data set to transform

    features : list or array-like
        Names of columns (variables) to transform using either one hot encoding or woe transformation

    target : string
        Name of the column with binary labels

    categories : list or array-like
        Names of categorical variables in the dataset. 


    """
    
    def __init__(self, data, features, target='target', categories=None):
        
        self.data = data
        
        self.features = features
        
        self.categories = categories
        
        self.target = target

        self.log = config.get_logger()

        if not os.path.isdir('data_prep'):
            os.mkdir('data_prep')
        
        self.binning_process = None
        
        self.encoder = None

        self.optb = None
        
    def create_encoder(self, drop='if_binary', verbose=False):

        """

        Create encode for one hot encoding

        Parameters
        ----------

        drop : string (default='if_binary', possible values {'first', 'if_binary'} or an array-like of shape (n_features,))  
            Specifies a methodology to use to drop one of the categories per feature.

        verbose : Boolean (default=False)
            Log to console or not.

        Returns
        -------
        
        None

        """

        start = int(time.time())
        
        X = self.data[self.categories].values
        
        self.encoder = OneHotEncoder(drop=drop, sparse=False)
        
        self.encoder = self.encoder.fit(X)

        end = int(time.time())

        if verbose:
            self.log.info('OneHotEncoder created')

        if verbose:
            self.log.info(f'Runtime for creating encoder: {int(end-start)} seconds')

        path = os.path.join(os.getcwd(), 'data_prep', 'encoder')

        with open(path, "wb") as file: 
            pickle.dump(self.encoder, file)

        if verbose:
            self.log.info(f'OneHot encoder saved: {path}')
        
    def oneHot_transform(self, data=pd.DataFrame(), verbose=False):

        """

        Transform data using one hot encoding

        Parameters
        ----------    
        data : pandas.DataFrame (default=empty dataframe)    
            Data to transfrom using one hot encoding

        Returns
        -------
        df_transformed : pandas.DataFrame
            Transformed dataset
            
        """
        
        if len(data) == 0:
            data = self.data

        X = data[self.categories].values

        if not self.encoder:
            self.create_encoder()

        start = int(time.time())

        X_encoded = self.encoder.transform(X)

        end = int(time.time())

        df_encoded = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names(self.categories))

        if verbose:
            self.log.info('Categorical variables succesful encoded')

        if verbose:
            self.log.info(f'Runtime for encoding categorical variables: {int(end-start)} seconds')

        df_transformed = pd.concat([data.reset_index(drop=True), df_encoded], axis=1)

        features = [col for col in self.features if col not in self.categories]

        self.features = list(set(features + list(df_encoded.columns)))

        if verbose:
            self.log.info(f'Total number of features after encoding: {len(self.features)}')
        
        self.df_transformed =  df_transformed

        
        return df_transformed
        
    def woe_bins(self, min_prebin_size = 0.1, 
                 selection_criteria = {"iv": {"min": 0.01, "max": 0.7, "strategy": "highest", "top": 50},
                                       "quality_score": {"min": 0.01}},
                 binning_fit_params = None,
                 verbose=False,
                 ):

        """

        Generate binning process for woe transformation
        
        Parameters
        ----------    
        
        min_prebin_size : float (default=0.1)
            The fraction of minimum number of records for each bin
            
        selection_criteria : dict or None 
            default
            -------
            {"iv": {"min": 0.001, "max": 0.7, "strategy": "highest", "top": 50}, 
            "quality_score": {"min": 0.001}}
            
            Variable selection criteria

        binning_fit_params : dict or None (default=None)
            Dictionary with optimal binning transform options for specific
            variables. Example ``{"variable_1": {"metric": "event_rate"}}``.
            
        Returns
        -------

        Self - Fitted binning process.
        
        """ 

        start = int(time.time())

        X = self.data[self.features]
        
        y = self.data[self.target]
        
        binning_process = BinningProcess(
            self.features,
            categorical_variables = self.categories,
            min_prebin_size = min_prebin_size,
            selection_criteria=selection_criteria,
            binning_fit_params = binning_fit_params
        )
        
        binning_process.fit(X, y)

        end = int(time.time())

        if verbose:
            self.log.info(f'Binning processor created')

        if verbose:
            self.log.info(f'Runtime for creating binning process : {int(end-start)} seconds')
        
        self.binning_process = binning_process

        path = os.path.join(os.getcwd(), 'data_prep', 'binningprocess.pkl')

        binning_process.save(path)

        if verbose:
            self.log.info(f'Binning processor saved : {path}')
    
    def woe_bin_table(self):

        """

        Generate summary results for the binning process
        
        Parameters
        ---------- 

        None
        
        Returns
        -------
        
        woe_table : pandas.DataFrame
            Summary results on the binning process
        
        """
        
        if not self.binning_process:
            self.woe_bins()
        
        woe_table = self.binning_process.summary()
        
        woe_table = woe_table.sort_values('iv', ascending=False).reset_index(drop=True)
        
        return woe_table
    
    def get_var_bins(self, var, ax, plot_type="WoE"):

        """

        Generate binning details for the variable
        
        Parameters
        ---------- 
        
        var : string 
            Name of variable for which to show binning tables

        ax : matplotlib.axes
            Axes to plot on.

        plot_type : string (default="WoE", options=[""WoE", "event_rate"])
            Whether to plot WoE or event rate per bin
            
        Returns
        -------   
        plot : matplotlib plot
        
        """

        if not self.binning_process:
            self.woe_bins()  
            
        self.optb = self.binning_process.get_binned_variable(var)

        table = self.optb.binning_table.build()
        
        table.drop(['Totals'], inplace=True)

        table['WoE'] = pd.to_numeric(table['WoE'])

        table = table.query("abs(WoE)>0")

        table['Bin'] = table['Bin'].astype('string')
        
        if plot_type == "WoE":

            sns.barplot(x='Bin', y='WoE', data=table, ax=ax, color='deepskyblue')

            ax.set_title('WoE per bins created \nfor {}'.format(var), fontsize = 12, fontweight='bold', fontname='Georgia')

            ax.tick_params(axis='both', which='major', labelsize=10)

            ax.tick_params(axis='x', which='major', rotation=90)

            ax.set_xlabel('Bin', fontsize=10)

            ax.set_ylabel('WoE', fontsize=10)

            ax.set_xlabel('Bin', fontsize=10)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
        
        else:
            
            ax.bar(table['Bin'], table['Event'], color='#ff2e63', label='Event')
            
            ax.bar(table['Bin'], table['Non-event'], bottom=table['Event'], color='#48d1cc', label='Non-event')
            
            ax.tick_params(axis='both', which='both', labelsize=10)
            
            ax.tick_params(axis='x', which='major', rotation=90)
            
            ax.legend(frameon=False, fontsize=8)
            
            ax.set_ylabel('Count', fontsize=10)

            ax.set_title('Event rate per bin \nfor {}'.format(var), fontname='Georgia', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Bin', fontsize=10)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            
            table['Bin'] = table['Bin'].astype('category')
            
            ax2 = ax.twinx()
            
            ax2.plot(table['Bin'], table['Event rate'], color='k', marker='o', label='event rate')
            
            ax2.set_yticklabels(['{:.0%}'.format(x) for x in ax2.get_yticks()])
            
            ax2.set_ylabel('Event rate', fontsize=10)

            ax2.grid(False)

            ax2.tick_params(axis='y', which='both', right=False, left=False)

            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
        
    def woe_transform(self, data=pd.DataFrame(), verbose=False):

        """

        Transform data using Weight of Evidence (WOE) weights
        
        Parameters
        ----------
        
        data : pandas.DataFrame (default=empty dataframe)
            Dataset to tranform

        verbose : boolean (default=False)
            Log to console or not.
            
        Returns
        -------

        pandas.DataFrame
            Transformed data.
        
        """
        
        if len(data) == 0:
            data = self.data
        
        if not self.binning_process:
            self.woe_bins()

        start = int(time.time())
        
        data_transf = data[self.features]

        data_transf = self.binning_process.transform(data_transf, metric="woe")
        
        df_transformed = pd.DataFrame(data_transf)
        
        df_transformed['target'] = data[self.target].values  

        end = int(time.time())

        if verbose:
            self.log.info('Data transformation completed')

        if verbose:
            self.log.info(f'Runtime for woe encoding : {int(end-start)} seconds')
        
        self.df_transformed = df_transformed
        
        return df_transformed
    
    def woe_features(self, verbose=False):
        
        """

        Generate variables selected using the selection criteria
        
        Parameters
        ----------
        None
        
        Returns
        -------

        features : list
            variables selected
        
        """
        
        if not self.binning_process:
            self.woe_bins()
            
        features = self.binning_process.get_support(names=True)
        
        if verbose:
            self.log.info(f'Number of features selected using the selection criteria defined : {len(features)} out of {len(self.features)}')
        
        return features
    
    def balance_data(self, target=None, data=pd.DataFrame()):
        
        """

        Balance data basing on each label size of the label variable.
        
        Parameters
        ----------
        
        data : pandas.DataFrame (default=empty dataframe)
            Dataset to balance.
            
        Returns
        -------

        pandas.DataFrame
            Transformed data.
            
        """

        if len(data) == 0:
            data = self.data

        if not target:
            target = self.target
        
        minority_size = data[target].value_counts()[1]

        self.balanced_df = data.groupby(target).sample(minority_size)
        
        self.balanced_df = self.balanced_df.sample(frac=1)
        
        return self.balanced_df
        