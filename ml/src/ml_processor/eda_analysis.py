
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import re
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

from ml_processor.configuration import config
from ml_processor.outliers import remove_outliers

def eda_data_quality(data, target=None):
    
    """

    Performing data quality checks on data set.
    
    Parameters
    ----------

    data : pandas.DataFrame 
        Data set to check.
        
    Returns
    -------

    df_results : pandas.DataFrame
        Results for quality checks

    """
    
    df_results = pd.DataFrame(data.dtypes)
    
    df_results.columns = ['type']
    
    df_results['type'] = df_results['type'].astype('string')
    
    df_uniq = pd.DataFrame(data.nunique())
    
    df_uniq.columns = ['unique']
    
    df_results = df_results.merge(df_uniq, how='left', left_index=True, right_index=True)
    
    missing = pd.DataFrame(data.isnull().sum())
    
    missing.columns = ['missing']
    
    df_results = df_results.merge(missing, how='left', left_index=True, right_index=True)
    
    df_results['pct.missing'] = (df_results['missing']/len(data))

    def get_mode(data):

        columns = data.columns.to_list()

        modes = {'column' : columns, 'mode' : [], 'pct.mode' : []}

        for col in columns:
            
            mode = data[col].value_counts(normalize=True).index[0]
            modes['mode'].append(mode)
            
            pct_mode = data[col].value_counts(normalize=True).values[0]
            modes['pct.mode'].append(pct_mode)
    
        mode_res = pd.DataFrame(modes)

        mode_res = mode_res.set_index('column')
        
        return mode_res
    
    df_modes = get_mode(data)

    df_results = df_results.merge(df_modes, how='left', left_index=True, right_index=True)
    
    df_summar = pd.DataFrame(data.describe()).T
    
    df_summar = df_summar[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    df_results = df_results.merge(df_summar, how='left', left_index=True, right_index=True)
    
    df_results = df_results.drop('std', axis=1)
    
    def check_dtype(x, y):
        if (any(re.findall(r'float|int', x, re.IGNORECASE)) and y < 30) or (x=='object' and y>30):
            return 45
        else:
            return 0
        
    df_results = (
        df_results
        .assign(rule_1 = np.where(df_results['pct.missing'] > 0.5, 80, 0))
        .assign(rule_2 = np.where(df_results['pct.missing'] > 0, 25, 0))
        .assign(rule_3 = np.where(df_results['min'] == df_results['75%'], 80, 0))
        .assign(rule_4 = np.where(df_results['min'] == df_results['50%'] , 60, 0))
        .assign(rule_4 = lambda X: np.where(X['50%'] == X['75%'] , 0, X['rule_4']))
        .assign(rule_5 = np.where(df_results['min'] < 0, 60, 0))
        .assign(rule_6 = list(map(check_dtype, df_results['type'], df_results['unique'])))
    )
    
    if target:
        df_results.loc[target, ['rule_3', 'rule_4', 'rule_6']] = 0

    df_results = (
        df_results
        .assign(quality_score = lambda X: 1 - ((X['rule_1'] + X['rule_2'] + X['rule_3'] + X['rule_4'] + X['rule_5'] + X['rule_6']) / 350))
    )
    
    cols = ['rule_1', 'rule_2', 'rule_3', 'rule_4', 'rule_5', 'rule_6']

    df_results[cols] = df_results[cols].where(df_results[cols]==0,1)
    
    df_results['pct.missing'] = df_results['pct.missing'].apply(lambda x: '{:.1%}'.format(x))
    
    log = config.get_logger()
    
    log.info(f'rule_1 : More than 50% of the data missing')
    log.info(f'rule_2 : Missing some data')
    log.info(f'rule_3 : 75% of the data is the same and equal to the minimum')
    log.info(f'rule_4 : 50% of the data is the same and equal to the minimum')
    log.info(f'rule_5 : Has negative values')
    log.info(f'rule_6 : Possible wrong data type')
    
    df_results = df_results.sort_values('quality_score', ascending=True)
    
    return df_results


class binary_eda_plot:
    
    """
    
    Visualizing data for explatory analysis.
    
    Parameters
    ----------
        
    data : pandas.DataFrame
        Data set for explatory analysis.
        
    plot_columns : dict (default=None) 
        Columns to visualize.
        
    log_columns : list (default=[None]) 
        Columns to use log scale.

    exclude_cols : list (default=[None]) 
        Columns to not to plot.
        
    columns : int (default=6)
        Number of columns in the matplotlib subplots.
    
    target_palette : dict (default = {1:'red', 0:'deepskyblue'}) 
        Palette for the labels.
    
    Returns
    -------
    
    plot: matplotlib plot
        
    """
    
    def __init__(self, data, 
                 target='target', 
                 plot_columns=None, 
                 log_columns=[None], 
                 exclude_cols=[None], 
                 columns=6, 
                 target_palette={1:'red', 0:'deepskyblue'},
                 bin_numeric=True,
                 sort_by_index=False
                ):
        
        self.data = data
        
        self.target = target
        
        self.exclude_cols = exclude_cols

        self.sort_by_index = sort_by_index

        if not plot_columns:
            
            self.plot_columns = {'target': [], 'discrete' : [], 'numeric': []}
            
            for col in self.data.columns:
            
                if col==self.target:
                    self.plot_columns['target']= col

                elif col not in self.exclude_cols:

                    if len(self.data[col].unique()) < 50 :
                        self.plot_columns['discrete'].append(col)

                    elif self.data[col].dtype in ('float', 'int'):
                        self.plot_columns['numeric'].append(col)
        else:
            self.plot_columns = plot_columns
        
        self.log_columns = log_columns
        
        # self.columns = columns
        
        self.target = target
        
        self.bin_numeric = bin_numeric
        
        self.target_palette = target_palette

        
        self.plot_vars = [self.plot_columns.get('target') ] + self.plot_columns.get('discrete') + self.plot_columns.get('numeric') 
        
        self.plot_vars = sorted(set(self.plot_vars), key=self.plot_vars.index)
        
        # double number of discrete variables for split between labels
        
        self.numb_vars = len(self.plot_vars) 
        self.columns = self.numb_vars
        
        # self.rows = math.ceil(self.numb_vars/self.columns)
        self.rows = 1
        
        self.length = self.rows * 6
        
        self.width = self.columns * 6
        
        
    def label_bar(self, axes, size):
        
        for ax in axes.patches:
            
            if ax.get_height()>0:
                
                value = '{:.0%}'.format(ax.get_height() / size)

                axes.text(ax.get_x() + (ax.get_width() * 0.5 )
                         ,ax.get_height() + (ax.get_height() * 0.025)
                         ,value
                         ,ha='center' 
                         ,va='center'
                         ,fontsize=10
                        )

                
    def format_ax(ax, colors = ['black', 'black', 'black', 'None']):
        
        ax.spines['left'].set_color(colors[0])
        
        ax.spines['left'].set_linewidth(1.5)
        
        ax.spines['bottom'].set_color(colors[1])
        
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.spines['right'].set_color(colors[1])
        
        ax.spines['right'].set_linewidth(1.5)
        
        ax.spines['top'].set_color(None)
  


    def plot_target(self, ax):
        
        sns.countplot(x=self.target, 
                      data=self.data, 
                      hue=self.target,
                      dodge=False, 
                      palette=self.target_palette, 
                      ax=ax
                     )
        
        plt.ticklabel_format(style='plain', axis='y')
        
        plt.title(self.target, fontsize=14, fontname='georgia', fontweight='bold')
        
        plt.xlabel('')
        
        self.label_bar(ax, len(self.data))
        
        plt.legend(loc='upper right')
        
    def create_bins(self, col):
        
        data = self.data[[col, self.target]]
        
#         # lower limit
#         bins = [data[col].min()]

#         qtiles = list(data[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values)

#         bins.extend(qtiles)
        
#         # upper limit
#         bins.append(data[col].max())
        
        bins = list(data[col].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).values)

#         pc_labels = [' (0%)', ' (10%()', ' (25%)', ' ( 50%)', ' (75%)', ' (90%)', ' (100%)']
        
        pc_labels = [f' ({i}%)' for i in range(0,110,10)]

        dic_Labels = dict(list(zip(bins, pc_labels)))

        labels = []

        uniq_bins = sorted(list(set(bins)))

        for i in range(1, len(uniq_bins)):
            
            bin_ = uniq_bins[i]
            
            label = 'upto_' + str(uniq_bins[i]) + dic_Labels[bin_]
            
            labels.append(label)
            
        bin_column = col + "_bin"
            
        data[bin_column] = pd.cut(data[col], bins=bins, include_lowest=True, duplicates='drop', labels=labels)
        
        return data
    

    def gen_plot_data(self, col, data=pd.DataFrame()):
        
        if len(data) == 0:
            data=self.data
        
        a = data[col].value_counts()
        if self.sort_by_index:
            a = a.sort_index()
        a.name = 'total'
        
        b = data[data[self.target]==1][col].value_counts()
        b.name = 'goods'
        
        c = data[data[self.target]==1][col].value_counts(normalize=True)
        c.name = 'distr_goods'
        
        d = data[data[self.target]==0][col].value_counts()
        d.name = 'bads'
        
        e = data[data[self.target]==0][col].value_counts(normalize=True)
        e.name = 'distr_bads'
        
        f = data.groupby(col)[self.target].mean()
        f.name = 'target_rate'

        g = a/len(data)
        g.name = 'attr_rate'

        results = pd.concat([a, b, c, d, e, f, g], axis=1)

        results.index = results.index.map(str)

        return results
    
        
    def plot_discrete(self, results, col, ax):
        
        ax.bar(results.index, results['goods'], color='#ff2e63')
        
        ax.bar(results.index, results['bads'], bottom=results['goods'], color='deepskyblue')
        
#         self.format_ax(ax)
        
        ax2 = ax.twinx()
        
        ax2.plot(results['distr_goods'], marker='o', color='#0000FF', label='event_rate')
        
        ax2.plot(results['distr_bads'], marker='o', color='black', label='non-event_rate')
        
        ax2.plot(results['target_rate'], marker='o', color='red', label='target_rate')

        # ax2.plot(results['attr_rate'], marker='o', color='#00FF00', label='attr_rate')
        
        ax2.set_yticklabels(['{:.0%}'.format(x) for x in ax2.get_yticks()])
        
        ax2.grid(False)
        
        ax.tick_params(axis='x', rotation=90)
        
        plt.title(col, fontsize=14, fontname='georgia', fontweight='bold')
        
        ax2.legend()
        
        plt.xlabel('')
        
        
    def plot_numeric(self, col, ax):
        
        data = remove_outliers(self.data, [col]).percentile_method(threshold=0.95)
        
        sns.kdeplot(
            data=data,
            x=col,
            hue='target',
            log_scale=True if col in self.log_columns else False,
            fill=True,
            palette=self.target_palette,
            ax=ax,
            legend = False
        )  
        
        plt.title(col, fontsize=14, fontname='georgia', fontweight='bold')
        
        plt.xlabel('')
        
        
    def get_plots(self):  

        fig2 = plt.figure(figsize=(self.width, self.length), constrained_layout=True)

        j = 0

        for n, col in enumerate(self.plot_vars):

            if self.plot_columns.get('target') == col:

                j += 1

                axes = plt.subplot(self.rows, self.columns, j)
                
                self.plot_target(axes)

            elif col in self.plot_columns.get('discrete'):

                j += 1

                ax1 = plt.subplot(self.rows, self.columns, j)
                
                results = self.gen_plot_data(col=col, data=pd.DataFrame())
                
                self.plot_discrete(results=results, col=col, ax=ax1)

            elif col in self.plot_columns.get('numeric'):

                j += 1

                ax2 = plt.subplot(self.rows, self.columns, j)
                
                if self.bin_numeric:
                    
                    data = self.create_bins(col)
                    
                    col_name = col + '_bin'
                    
                    results = self.gen_plot_data(col=col_name, data=data)
                    
                    self.plot_discrete(results=results, col=col_name, ax=ax2)
                
                else:
                    
                    self.plot_numeric(col, ax2)

        plt.tight_layout()
