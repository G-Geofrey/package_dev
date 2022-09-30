from scipy import stats
import copy

from ml_processor.configuration import config 

class remove_outliers:

    """

    Removing outliers from data.
    
    Parameters
    ----------
        
    data : pandas.DataFrame
        Data set from which to remove outliers

    columns : list or arraay-like 
        Names of columns from which to remove outliers
        
    split :boolean (default=True) 
        Whether to remove outliers according to different classes of a target column
        
    target : string (default='target') 
        Column name with classes for removing outliers separately if split=True

    log : boolean (default=False)
        Whether to print out information on the results

    """
    def __init__(self, data, columns, split=True, target='target', log=False):
        
        self.data = data
        
        self.columns = columns
        
        self.split = split
        
        self.target = target
        
        self.log=log
        
        self.logger = config.get_logger()
    
    def std_method(self, threshold=3):

        """

        Remove outliers from data set using the z-score method.
        
        Parameters
        ----------
        
        threshold : int (default=3) 
            z-score from which the value is an outlier
            
        Returns
        -------
        
        pandas.DataFrame: 
            Dataset of original dataset without outliers

        """
        
        data = self.data.copy()
        
        if self.split:
            
            for col in self.columns:
                
                data = data.groupby(self.target).apply(lambda x: x[(abs(stats.zscore(x[col])) < threshold)])
                
                data.index = data.index.droplevel(0)
            
            if self.log:
                
                labels = self.data[self.target].unique()
                
                for col in self.columns:
                    
                    mask = []
                    
                    for label in labels:
                        
                        check_data = self.data[self.data[self.target]==label]
                        
                        rm_index = list(check_data[(np.abs(stats.zscore(check_data[col])) >= threshold)].index)
                        
                        mask.extend(rm_index)
                        
                        rows_dropped = len(rm_index)
                        
                        self.logger.info(f'Number of rows dropped for column={col} and target_label={label}: {rows_dropped}')
                        
                        floor = self.data.loc[~self.data.index.isin(rm_index)][col].min()
                        
                        ceiling = self.data.loc[~self.data.index.isin(rm_index)][col].max()
                        
                        self.logger.info(f'Data range for column={col} and target_label={label}: {[floor, ceiling]}')
                    
                    total_rows_dropped = len(set(mask))
                    
                    pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))
                    
                    self.logger.info(f'Number of rows dropped for column={col}: {total_rows_dropped} ({pct_dropped})')
        else:
            
            mask = []
            
            for col in self.columns:
                
                data = data[(abs(stats.zscore(data[col])) < threshold)]
            
            if self.log:
                
                for col in self.columns:
                    
                    rm_index = list(self.data[(abs(stats.zscore(self.data[col])) < threshold)].index)
                    
                    total_rows_dropped = len(self.data) - len(rm_index)
                    
                    pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))
                    
                    self.logger.info(f'Number of rows dropped for column={col}: {total_rows_dropped} ({pct_dropped})')
    
        if self.log:
            
            total_rows_dropped = len(self.data) - len(data)
            
            pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))
            
            self.logger.info(f'Total number of rows dropped: {total_rows_dropped} ({pct_dropped})')

        return data
    
    def percentile_method(self, threshold=0.95):

        """

        Remove outliers from data set using the percentile method.
        
        Parameters
        ----------
        
        threshold : int (default=0.95) 
            Percentile from which the value is an outlier
            
        Returns
        -------
        
        pandas.DataFrame: 
            Dataset of original dataset without outliers

        """
        
        data = self.data.copy()
        
        if self.split:
            
            columns = self.columns + [self.target]
            
            quantiles = self.data[columns].groupby(self.target).quantile(threshold)
            
            mask_all = []
            
            for col in self.columns:
                
                label_qtiles = quantiles[col].to_dict()
                
                mask = []
                
                for label in label_qtiles:
                    
                    check_data = self.data[(self.data[self.target]==label)]
                    
                    rm_index = list(check_data[check_data[col] >= label_qtiles[label]].index)
                    
                    mask.extend(rm_index)
                    
                    rows_dropped = len(rm_index)
                    
                    if self.log:
                        
                        self.logger.info(f'Number of rows dropped for column={col} and target_label={label}: {rows_dropped}')
                        
                        self.logger.info(f'Ceiling for column={col} and target_label={label}: {label_qtiles[label]}')
                
                if self.log:
                    
                    total_rows_dropped = len(set(mask))
                    
                    pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))
                    
                    self.logger.info(f'Number of rows dropped for column={col}: {total_rows_dropped} ({pct_dropped})')             
                
                mask_all.extend(mask) 
            
            data = data.loc[~data.index.isin(mask_all)]
            
            if self.log:

                total_rows_dropped = len(set(mask_all))

                pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))

                self.logger.info(f'Total number of rows dropped: {total_rows_dropped} ({pct_dropped})')

        else:
            
            quantiles = self.data[self.columns].quantile(threshold)
            
            cut_offs = zip(quantiles.index, quantiles.values)
            
            mask_all = []
            
            for (col, cut_off) in cut_offs:
                
                rm_index = list(self.data[self.data[col] >= cut_off].index)
                
                mask_all.extend(rm_index)
                
                if self.log:
                    
                    rows_dropped = len(rm_index)
                    
                    pct_dropped = '{:.1%}'.format(rows_dropped/len(self.data))
                    
                    self.logger.info(f'Number of rows dropped for column={col}: {rows_dropped} ({pct_dropped})')
                    
            data = self.data.loc[~self.data.index.isin(mask_all)]
        
            if self.log:

                total_rows_dropped = len(set(mask_all))

                pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))

                self.logger.info(f'Total number of rows dropped: {total_rows_dropped} ({pct_dropped})')
            
        return data
    
    
    def iqr_method(self):

        """

        Remove outliers from data set using the inter-quantile method.
        
        Parameters
        ----------
        
        None
            
        Returns
        -------
        
        pandas.DataFrame: 
            Dataset of original dataset without outliers

        """

        data = self.data.copy()
        
        if self.split:
            
            mask_all = []
            
            for col in self.columns:
                
                mask = []
                
                labels = self.data[self.target].unique()
                
                for label in labels:
                    
                    check_data = self.data[(self.data[self.target]==label)]

                    Q1, Q3 = check_data[col].quantile([0.25, 0.75])
                    
                    IQR = Q3 - Q1
                    
                    ll = Q1 - (1.5 * IQR)
                    
                    ul = Q1 + (1.5 * IQR)

                    rm_index = list(check_data[(check_data[col] >= ul) | (check_data[col] <= ll)].index)
                    
                    mask.extend(rm_index)
                    
                    if self.log:

                        rows_dropped = len(rm_index)

                        pct_dropped = '{:.1%}'.format(rows_dropped/len(self.data))

                        self.logger.info(f'Number of rows dropped for column={col}: {rows_dropped} ({pct_dropped})')
                        
                        self.logger.info(f'Data range for column={col} and target_label={label}: {[ll, ul]}')
                
                mask_all.extend(mask)
                
                if self.log:
                    
                    total_rows_dropped = len(set(mask))
                    
                    pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))
                    
                    self.logger.info(f'Number of rows dropped for column={col}: {total_rows_dropped} ({pct_dropped})')  
            
            data = self.data.loc[~self.data.index.isin(mask_all)]
            
            if self.log:

                total_rows_dropped = len(set(mask_all))

                pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))

                self.logger.info(f'Total number of rows dropped: {total_rows_dropped} ({pct_dropped})')
        
        else:
            
            mask_all = []
            
            for col in self.columns:
                
                Q1, Q3 = self.data[col].quantile([0.25, 0.75])
                
                IQR = Q3 - Q1
                
                ll = Q1 - (1.5 * IQR)
                
                ul = Q1 + (1.5 * IQR)
                
                rm_index = list(self.data[(self.data[col] >= ul) | (self.data[col] <= ll)].index)
                
                mask_all.extend(rm_index)
                
                if self.log:
                    
                    rows_dropped = len(rm_index)

                    pct_dropped = '{:.1%}'.format(rows_dropped/len(self.data))

                    self.logger.info(f'Number of rows dropped for column={col}: {rows_dropped} ({pct_dropped})')

                    self.logger.info(f'Data range for column={col}: {[ll, ul]}')
            
            data = self.data.loc[~self.data.index.isin(mask_all)]
            
            if self.log:

                total_rows_dropped = len(set(mask_all))

                pct_dropped = '{:.1%}'.format(total_rows_dropped/len(self.data))

                self.logger.info(f'Total number of rows dropped: {total_rows_dropped} ({pct_dropped})')

        return data
