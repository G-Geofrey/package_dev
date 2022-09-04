import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import warnings

warnings.filterwarnings('ignore')


from ml_processor.outliers import remove_outliers

def eda_data_quality (data):
    
    """
    Function for performing data quality checks on data set
    
    Args:
        data (dataframe) data on which checks are to be performed
        
    Returns:
        dataframe: results for quality checks
    """
    
    df_results = pd.DataFrame(data.dtypes)
    
    df_results.columns = ['type']
    
    df_uniq = pd.DataFrame(data.nunique())
    
    df_uniq.columns = ['unique']
    
    df_results = df_results.merge(df_uniq, how='left', left_index=True, right_index=True)
    
    missing = pd.DataFrame(data.isnull().sum())
    
    missing.columns = ['missing']
    
    df_results = df_results.merge(missing, how='left', left_index=True, right_index=True)
    
    df_results['pct.missing'] = (df_results['missing']/len(data)).apply(lambda x: '{:.1%}'.format(x))
    
    df_summar = pd.DataFrame(data.describe()).T
    
    df_summar = df_summar[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    df_results = df_results.merge(df_summar, how='left', left_index=True, right_index=True)
    
    df_skew = pd.DataFrame(data.skew())
    
    df_skew.columns = ['skewness']
    
    df_results = df_results.merge(df_skew, how='left', left_index=True, right_index=True)
    
    df_kurtosis = pd.DataFrame(data.kurtosis())
    
    df_kurtosis.columns = ['kurtosis']
    
    df_results = df_results.merge(df_kurtosis, how='left', left_index=True, right_index=True)
    
    return df_results


class binary_eda_plot:
    
    """
    Class for visualizing data fro explatory analysis
    
    Attributes:
        data (dataframe) data for explatory analysis
        plot_columns (dic) columns to visualize
        log_columns (list) columns to use log scale
        columns (int) number of columns in the matplotlib subplots
        target_palette (dic) palette for the labels
        
    """
    
    def __init__(self, data, plot_columns, log_columns=[None], columns=9, target_palette={1:'deepskyblue', 0:'red'}):
        
        self.data = data
        
        self.plot_columns = plot_columns
        
        self.log_columns = log_columns
        
        self.columns = columns
        
        self.target = self.plot_columns.get('target')
        
        self.target_palette = target_palette
        
        self.plot_vars = [self.plot_columns.get('target') ] + self.plot_columns.get('discrete') + self.plot_columns.get('numeric') 
        
        self.plot_vars = sorted(set(self.plot_vars), key=self.plot_vars.index)
        
        # double number of discrete variables for split between labels
        
        self.numb_vars = len(self.plot_vars) + len(self.plot_columns.get('discrete')) 
        
        self.rows = math.ceil(self.numb_vars/self.columns)
        
        self.length = self.rows * 4
        
        self.width = self.columns * 6
        
    def label_bar(self, axes, size):
        
        for ax in axes.patches:
            
            value = '{:.0%}'.format(ax.get_height() / size)
            
            axes.text(ax.get_x() + (ax.get_width() * 0.5 )
                     ,ax.get_height() + (ax.get_height() * 0.025)
                     ,value
                     ,ha='center' 
                     ,va='center'
                     ,fontsize=10
                    )
        
    def plot_target(self, ax):
        
        sns.countplot(x=self.target, data=self.data, palette=self.target_palette, ax=ax)
        
        plt.ticklabel_format(style='plain', axis='y')
        
        plt.title(self.target)
        
        plt.xlabel('')
        
        self.label_bar(ax, len(self.data))
        
    def plot_discrete(self, col, ax, target_index):
        
        data = self.data[self.data[self.target]==target_index]
        
        sns.countplot(x=col, data=data, ax=ax, color=self.target_palette[target_index])
        
        plt.title(f'{col} : label = {target_index}')
        
        plt.tick_params(axis='x', which='major', labelsize=8)
        
        plt.xlabel('')
        
    def plot_numeric(self, col, ax, j):
        
        data = remove_outliers(self.data, [col], target=self.target)
        
        sns.kdeplot(
            data= self.data,
            x=col,
            hue=self.target,
            log_scale=True if col in self.log_columns else False,
            fill=True,
            palette=self.target_palette,
            ax=ax,
            legend = False if j < (self.numb_vars) else True
        )  
        
        plt.title(col)
        
        plt.xlabel('')
        
    def get_plots(self):  

        fig2 = plt.figure(figsize=(self.width, self.length))

        j = 0

        for n, col in enumerate(self.plot_vars):

            if self.plot_columns.get('target') == col:

                j += 1

                axes = plt.subplot(self.rows, self.columns, j)
                
                self.plot_target(axes)

            elif col in self.plot_columns.get('discrete'):

                for i in range(2):

                    j += 1

                    ax = plt.subplot(self.rows, self.columns, j)
                    
                    self.plot_discrete(col, ax, i)

            elif col in self.plot_columns.get('numeric'):

                j += 1


                ax = plt.subplot(self.rows, self.columns, j)
                
                self.plot_numeric(col, ax, j=j)

        plt.subplots_adjust(wspace=0.4, hspace=0.25)

        plt.show()
