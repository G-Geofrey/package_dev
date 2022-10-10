
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
import os
import json
import joblib
import time

from datetime import datetime
from collections import defaultdict

import xgboost as xgb
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, space_eval
from sklearn import metrics

from ml_processor.configuration import config
from ml_processor.jsonSerializer import NpEncoder

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')


class xgbmodel:
    
    """

    Performing machine learning tasks including hyperparameter tuning and xgb model fitting.
    
    Parameters
    ----------
        
    df : pandas.Dataframe 
        Dataset with features and labels.

    features : list or array-like 
        Variable names (features) for fitting the model.

    target : string 
        Name of column with labels (dependent variable).

    params_prop : float (default=0.25) 
        Proportion of data set to use for hyperparameter tuning.

    test_size : float (default=0.33) 
        Proportion of data to use as the test set.

    hyperparams : dictionary (default=None) 
        Predefined hyperparameters and their values.Specified if hyperparameter tunning is not necessary.
    
    scoring : string
        Performance metric to maximises.
    
    """

    
    def __init__(self, df, features, target, params_prop=0.25, test_size=0.33, hyperparams=None, scoring='recall'):
        
        self.data = df
        
        self.features = features
        
        self.target = target
        
        self.hyperparams = hyperparams

        self.scoring = scoring
        
        self.cwd = os.getcwd()
        
        self.log = config.get_logger(file='xgbModelLogs.log')
        
        if not os.path.isdir('model_artefacts'):
            os.mkdir('model_artefacts')

        self.path = os.path.join('model_artefacts', 'xgbModel_results.json')
        
        try:
            with open(self.path) as file:
                self.model_artefacts = json.load(file)
        except:
            self.model_artefacts = {}

        self.model_name = 'xgbmodel_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
        
        self.log.info(f'Model job name: {self.model_name}')
        
        self.model_artefacts[self.model_name] = {}
        
        self.params_prop = params_prop
        self.model_artefacts[self.model_name]['tunning_data_size'] = self.params_prop
        
        self.test_size = test_size
        self.model_artefacts[self.model_name]['test_data_size'] = self.test_size
    
        
    def split_data(self): 
        
        """

        Split data into training and test sets

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        X = self.data.drop(self.target, axis=1)
        
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)
        
        self.log.info('Splitting data into training and testing sets completed')
        
        self.log.info(f'Training data set:{self.X_train.shape[0]} rows')
        
        self.log.info(f'Testing data set:{self.X_test.shape[0]} rows')


    def reduce_data(self):
        
        """

        Generate a fraction of the data to use for hyperparameter tuning data 

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        pandas.DataFrame:
            Dataset for hyperparameter tuning
        
        """
        
        tunning_data_size = int(self.data.shape[0] * self.params_prop)
        
        self.log.info('Hyper parameter tuning data set created')
    
        self.log.info(f'Hyper parameter tuning data set:{tunning_data_size} rows')
        
        return self.data.sample(tunning_data_size)

        
    def split_tunning_data(self):
        
        """

        Split data into training and test sets

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        tunning_data = self.reduce_data()
        
        X = tunning_data.drop(self.target, axis=1)
        
        y = tunning_data[self.target]

        self.X_train_param, self.X_test_param, self.y_train_param, self.y_test_param = train_test_split(X, y, test_size=self.test_size)
        
        self.log.info('Splitting hyperparameter tuning data into training and testing sets completed')
        
        self.log.info(f'Hyperparameter tuning training data set:{self.X_train_param.shape[0]} rows')
        
        self.log.info(f'Hyperparameter tuning testing data set:{self.X_test_param.shape[0]} rows')
        

    def hyper_parameter_tunning(self):
        
        """

        Hyperparameter tuning using hyperopt method.

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        self.split_tunning_data()
        
        start = time.process_time()
  
        space = {
            'learning_rate': hp.choice('learning_rate', [0.0001,0.001, 0.01, 0.1, 1]),
            'max_depth': hp.choice('max_depth', range(3, 13, 2)),
            'gamma': hp.choice('gamma', np.arange(0.1, 0.6, 0.1)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 1.0, 0.1)),
            'reg_alpha': hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
            'reg_lambda': hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100]),
        }
        
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        def objective_func(params):
            
                xgboost = xgb.XGBClassifier(seed=0, **params)
    
                scores = cross_val_score(
                    xgboost,
                    self.X_train_param[self.features], 
                    self.y_train_param,
                    cv=kfold, 
                    scoring='recall', 
                    n_jobs=-1
                )

                best_score = max(scores)

                loss = - best_score

                return {'loss':loss, 'params':params, 'status':STATUS_OK}
        
        self.log.info('Trials initialized...')
        
        trials = Trials()

        param_search = fmin(fn = objective_func, space=space, algo=tpe.suggest, max_evals=48, trials=trials)
        
        end = time.process_time()
        
        self.log.info('Hyperparameter tuning completed')
        
        self.log.info(f'Runtime for Hyperparameter tuning : {int(end-start)} seconds')
        
        self.hyperparams = space_eval(space, param_search)
        
        self.log.info(f'Best parameters: {self.hyperparams}')
        
        self.model_artefacts[self.model_name]['best_params'] = self.hyperparams
        
    def fit_model(self):
        
        """

        Fit xgb model regression model from xgboost module
        
        Parameters
        ---------- 
        
        None
        
        Returns
        -------
        
        model : object
            xgboost model object
        
        """
        
        self.split_data()
        
        if not self.hyperparams:
            self.hyper_parameter_tunning()

        best_params = self.hyperparams
        
        self.log.info('Model fitting initialized...')
        
        start = time.process_time()

        xgb_model = xgb.XGBClassifier(seed=0, **best_params)
        
        self.log.info('Model fitting started...')

        xgb_model.fit(self.X_train[self.features], self.y_train)
        
        end = time.process_time()
        
        self.log.info('Model fitting completed')
        
        self.log.info(f'Runtime for fitting the model : {int(end-start)} seconds')
        
        self.model = xgb_model

        path = os.path.join(os.getcwd(), 'model_artefacts', str(self.model_name) + '.sav')

        joblib.dump(xgb_model, path)

        self.log.info(f'Model saved: {path}')

        
    def get_feature_imp(self):
        
        """

        Function for generating feature importance for fitted model 
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        self.df_features = pd.DataFrame({'feature' : self.features, 'importance' : self.model.feature_importances_})

        self.df_features = self.df_features.sort_values(by='importance', ascending=False)
        
        self.log.info('Dataframe with feature importance generated')
        
        self.model_artefacts[self.model_name]['feature_importance'] = {'feature' : list(self.features), 'importance' : list(self.model.feature_importances_)}
    
    
    def get_metrics(self):
        
        """

        Function for generating feature model performance metrices for fitted model 

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        self.get_feature_imp()
        
        self.pred_class = self.model.predict(self.X_test[self.features])

        self.log.info('Predicted labels generated (test)')
        

        self.pred_prob = self.model.predict_proba(self.X_test[self.features])[:,1]

        self.log.info('Predicted probabilities generated (test)')

        self.confMatrix = metrics.confusion_matrix(self.y_test, self.pred_class)

        self.log.info('Confusion matrix generated (test)')

        self.area_roc = metrics.roc_auc_score(self.y_test, self.pred_prob)
        
        self.model_artefacts[self.model_name]['area_roc'] = float(self.area_roc)
        
        self.log.info("AUC (test): {:.0%}".format(self.area_roc))

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.y_test, self.pred_class)
        
        self.model_artefacts[self.model_name]['precision'] = float(precision[1])
        
        self.log.info("Precision (test): {:.0%}".format(precision[1]))

        self.model_artefacts[self.model_name]['recall'] = float(precision[1])
        
        self.log.info("Recall (test): {:.0%}".format(recall[1]))

        self.model_artefacts[self.model_name]['fscore'] = fscore[1]
        
        self.log.info("F_score (test): {:.0%}".format(fscore[1]))

        self.precision_curve, self.recall_curve, _ = metrics.precision_recall_curve(self.y_test, self.pred_prob)

        self.log.info('Precision and Recall values for the precision recall curve created')


        self.trP, self.trN, _ = metrics.roc_curve(self.y_test, self.pred_prob)

        self.log.info('True positive and negativevalues for the ROC curve created')

        
        perf_diff_thresholds = {'thresholds':[], 'precision':[], 'recall':[], 'f1_score':[] }
        
        perf_diff_thresholds['thresholds'] = np.arange(0, 1+1e-5, 0.05)
        
        for i in perf_diff_thresholds['thresholds']:
            
            precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
                self.y_test, self.pred_prob > i, average='binary')
            
            perf_diff_thresholds['precision'].append(precision)
            
            perf_diff_thresholds['recall'].append(recall)
            
            perf_diff_thresholds['f1_score'].append(f1_score)
        
        perf_diff_thresholds =  pd.DataFrame(perf_diff_thresholds)
        
        self.log.info('Recall and precision calculation for different thresholds (test) completed')
        
        self.perf_diff_thresholds = perf_diff_thresholds
    
    def model_results(self):

        """

        Fitting model and performing model diagnostics.

        Parameters
        ---------- 
        
        None
        
        Returns
        -------
        
        object:
            xgb model object.
        
        """
        
        self.fit_model()
        
        self.get_metrics()
        
        # saving all model artefacts
        json_object = json.dumps(self.model_artefacts, cls=NpEncoder, indent=4)

        with open(self.path, 'w') as outfile:
            outfile.write(json_object)
        
        
        return self.model
        
    def format_grid(ax):
        
        xgrid_lines = ax.get_xgridlines()
        
        n_xgrid = xgrid_lines[1]
        
        n_xgrid.set_color('k')
        
        ygrid_lines = ax.get_ygridlines()
        
        n_ygrid = ygrid_lines[1]
        
        n_ygrid.set_color('k')
        
    def plot_comf_matrix(self, cm, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
            
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 12}, ax=ax)
        
        ax.set_xlabel('Predicted label', fontsize=12)
        
        ax.set_ylabel('True label', fontsize=12)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_frame_on(False)
        
        ax.set_title('Confusion Matrix', fontsize=12)
        
    def plot_roc_curve(self, trP, trN, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
            
        ax.plot(trP, trN, color='orange')
        
        ax.plot([1,0], [1,0], "--", color='darkblue')
        
        ax.set_frame_on(False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        
        ax.set_ylabel('True Positive Rate', fontsize=12)
        
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12)
        
        ax.text(0.3, 0.7, 'AUC = {:.3f}'.format(self.area_roc), fontsize=12)
        
        # self.format_grid(ax)
        
    def plot_rp_curve(self, rcl, prc, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
            
        ax.plot(rcl, prc, color='orange')
        
        ax.set_frame_on(False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_xlabel('recall', fontsize=12)
        
        ax.set_ylabel('precision', fontsize=12)
        
        ax.set_title('Precision recall Curve', fontsize=12)
        
        # self.format_grid(ax)
        
    def plot_prediction(self, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
        
        palette = {0:'deeppink', 1:'green'}
        
        df_pred = pd.DataFrame({'actual_class': self.y_test, 'pred_class': self.pred_class, 'pred_prob': self.pred_prob})
        
        sns.kdeplot(data=df_pred, x='pred_prob', hue='pred_class', fill=True, legend=False, palette=palette, ax=ax)
        
        ax.set_facecolor('#274472')
        
        ax.set_xlabel('Probability of event', fontsize=12, color='white')
        
        ax.set_ylabel('Probability Density', fontsize=12)
        
        ax.set_title('Prediction distribution', fontsize=12)
        
        ax.tick_params(axis='both', which='major')
        
        ax.grid(False)
        
    def plot_feature_imp(self, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
        
        sns.barplot(x='feature', y='importance', data=self.df_features, palette='RdBu_r', ax=ax) 
        
        ax.tick_params(axis='x', rotation=90)
        
        ax.set_frame_on(False)
        
        ax.set_xlabel('Feature', fontsize=12)
        
        ax.set_ylabel('Feature importance', fontsize=12)
        
        ax.set_title('Features importance', fontsize=12)

    def make_plots(self):
        
        if not self.model:
            log.error('Fit model first using model_results method')
        else:

            cm = self.confMatrix

            area_roc = self.model_artefacts[self.model_name]['area_roc']


            trP = self.trP

            trN = self.trN


            fig = plt.figure(figsize=(20,10), constrained_layout=True)

            spec2 = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

            fig_ax1 = fig.add_subplot(spec2[0,0])
            fig_ax2 = fig.add_subplot(spec2[0,1])
            fig_ax3 = fig.add_subplot(spec2[0,2])
            fig_ax4 = fig.add_subplot(spec2[0,3])
            fig_ax5 = fig.add_subplot(spec2[1,:])

            # confusion matrix
            self.plot_comf_matrix(cm, ax=fig_ax1)

            self.plot_roc_curve(trP, trN, ax=fig_ax2)

            self.plot_rp_curve(self.recall_curve, self.precision_curve, ax=fig_ax3)

            self.plot_prediction(ax=fig_ax4)

            self.plot_feature_imp(ax=fig_ax5)

            plt.subplots_adjust(hspace=0.3, wspace=0.3)

            plt.show()
            
#################################################################################

class logitmodel:
    
    """

    process data and fit logistic model from the python statsmodel.formula.api module.
    
    Parameters
    ----------
    
    data : pandas.DataFrame
        Data to process and use for fitting model
        
    features : list or array-like
        Name of variable columns 
        
    target : string (default='target')
        Name of label column
    
    test_size : float (default=0.33)
        Size of test data set, for splitting data between training ans test sets.
        
    rm_corr_vars : boolen (default=True)
        Whether to drop correlated variables or not. If two or more variables are correlated, 
        the variable with the highest IV value is retained while the others are dropped
        
    iv_df : pandas.DataFrame (default=empty dataframe)
        Binning process summary results
        
    threshold : float (default=0.5)
        Absolute correlation value from which correlated variables are dropped
        
    significant : boolen (default=True)
        Whether to include on significant variables in the logistic model. If true, variables 
        are selected using forward elimination
        
    """
    
    def __init__(self, data, features, target='target', test_size=0.33, rm_corr_vars=True, iv_df=None, 
                 threshold=0.5, significant=True):
        
        self.data = data
        
        self.features = features
        
        self.target = target
        
        self.test_size = test_size
        
        self.rm_corr_vars = rm_corr_vars
        
        self.iv_df = iv_df
        
        self.significant = significant
        
        self.best_features = None
        
        self.threshold = threshold

        self.log = config.get_logger(file='logitModelLogs.log')

        if not os.path.isdir('model_artefacts'):
            os.mkdir('model_artefacts')

        self.path = os.path.join('model_artefacts', 'logitModel_results.json')

        try:
            with open(self.path) as file:
                self.model_artefacts = json.load(file)
        except:
            self.model_artefacts = {}

        self.model_name = 'logitModel_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))

        self.log.info(f'Model job name: {self.model_name}')
        
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
        self.full_train_data = pd.DataFrame()
        
        self.model = None
        
        self.pred_prob = []
        
#         self.uniq_val_count = pd.Series()
        
        self.dropped_vars = defaultdict(list)

        self.metrics = False
    
    def drop_singular_vars(self, data=pd.DataFrame()):
        
        """

        Drop variables which have only a single value
        
        Parameters
        ----------
        
        data : pandas.DataFrame (default=empty dataframe)
            Dataset with columns to drop
        
        Returns
        -------
        
        data : pandas.DataFrame
            Data after single value columns dropped
            
        """
        
        if len(data) == 0:
            data = self.data

        start = time.process_time()
        
        self.uniq_val_count = data[self.features].nunique()
        
        drop_features = list(self.uniq_val_count[self.uniq_val_count==1].index)
        
        data = data.drop(drop_features, axis=1)

        end = time.process_time()
        
        self.features = list(set(self.features) - set(drop_features))
        
        self.log.info(f'Single value columns dropped : {drop_features}')

        self.log.info(f'Runtime for dropping single value columns : {int(end-start)} seconds')
        
        return data
    
    def rm_correlation(self, iv_df=pd.DataFrame()):
        
        """

        Remove highly correlated variables from list of variables
        
        Parameters
        ----------
        
        iv_df : pandas.DataFrame (default=empty dataframe)
            Binning process summary results
            
        Returns
        -------
        
        None
        
        """
        
        if len(iv_df) == 0:
            iv_df = self.iv_df

        start = time.process_time()

        self.corr_res = abs(self.data[self.features].corr())
        
        self.iv_res = iv_df[['name','iv']].set_index('name')['iv']
        
        woe_features = []

        for col in self.features:

            # get variables that are highly correlated with selceted variable

            corr_arr = self.corr_res[col].index[self.corr_res[col].values > self.threshold]

            # by default correlation between column and itself is 1, remove it from correlated variables

            corr_arr = list(set(corr_arr) - {col})

            if corr_arr:

                # get IV values for highly correlated variables

                corr_Vars_iv = self.iv_res[corr_arr].values

                # index of variables with higher IV

                iv_index = corr_Vars_iv > self.iv_res[col]

                # variables with higher IV value

                higher_iv_vars = list(self.iv_res[corr_arr].index[iv_index])

                if higher_iv_vars:

                    for var in higher_iv_vars:

                        self.dropped_vars['var'].append(col)

                        self.dropped_vars['corr_var'].append(var)

                        self.dropped_vars['corr'].append(self.corr_res.loc[col, var])

                        self.dropped_vars['var_iv'].append(self.iv_res[col])

                        self.dropped_vars['corr_var_iv'].append(self.iv_res[var])

                else:

                    woe_features.append(col)
            else:

                woe_features.append(col)
        
        self.features  = woe_features 

        end = time.process_time()

        self.log.info('Dropping correlated variables completed.')
        
        self.log.info(f'Number of variables dropped due to correlation : {len(self.features) - len(woe_features)}')

        self.log.info(f'Runtime for dropping correlated variables : {int(end-start)} seconds')

        
    def get_dropped_vars(self):
        
        """

        Get all correlated variables dropped
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        pandas.DataFrame:
            Dropped variables and correlated variables
        
        """
        
        if len(self.dropped_vars) == 0:
            self.rm_correlation()
            
        return pd.DataFrame(self.dropped_vars)
        
    def split_data(self):
        
        """

        Split data into training and test sets

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        self.data = self.drop_singular_vars()
        
        if self.rm_corr_vars:
            self.rm_correlation()
        
        X = self.data.drop(self.target, axis=1)
        
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)
        
        self.full_train_data = self.X_train
        
        self.full_train_data = pd.concat([self.full_train_data, self.y_train], axis=1)
        
        self.log.info('Splitting data into training and testing sets completed')
        
        self.log.info(f'Training data set : {self.X_train.shape[0]} rows')
        
        self.log.info(f'Testing data set : {self.X_test.shape[0]} rows')
    
    def get_features(self):
        
        """

        Get significant variables with logistic regression
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        if len(self.full_train_data)==0:
            self.split_data()

        start = time.process_time()
            

        best_features = []

        self.log.info('Feature selection initiated...')

        while (len(self.features)>0):

            remaining_features = list(set(self.features) - set(best_features))

            new_pval = pd.Series(index=remaining_features)

            for new_column in remaining_features:

                cols = best_features + [new_column]

                formular = self.target + ' ~ ' + (' + '.join(cols))
                
                model = smf.logit(formular, data = self.full_train_data).fit(disp=0, method='bfgs')


                new_pval[new_column] = model.pvalues[new_column]

            min_p_value = new_pval.min()

            if(min_p_value<0.05):

                best_features.append(new_pval.idxmin())

            else:

                break

        end = time.process_time()

        self.log.info('Feature selection completed')

        self.log.info(f'Number of features dropped: {len(self.features) - len(best_features)}')

        self.log.info(f'Runtime for feature selection : {int(end-start)} seconds')
        
        self.best_features = best_features
    
    def fit_model(self):
        
        """

        Fit logistic regression model from python statsmodel.formula.api module
        
        Parameters
        ---------- 
        
        None
        
        Returns
        -------
        
        model : object
            statsmodel.formual.api model object
        
        """
        
        if len(self.full_train_data)==0:
            self.split_data()
        
        if self.significant:
            if not self.best_features:
                self.get_features()
                
            self.used_features = self.best_features
            
        else:
            
            self.used_features = self.features


        json_object = json.dumps(self.used_features, indent=4)

        path = os.path.join(os.getcwd(), 'model_artefacts', str(self.model_name) + 'features.json')

        with open(path, 'w')  as file:
            file.write(json_object)

        self.log.info(f'Features saved: {path}')

        start = time.process_time()
        
        formular = self.target + ' ~ ' + (' + '.join(self.used_features))

        self.log.info(f'Model fitting initiated...')
        
        self.model = smf.logit(formular, data = self.full_train_data).fit(disp=0, method='bfgs')

        end = time.process_time()

        self.log.info(f'Model fitting completed')

        self.log.info(f'Runtime for fitting the model : {int(end-start)} seconds')

        path = os.path.join(os.getcwd(), 'model_artefacts', self.model_name + '.sav')

        joblib.dump(self.model, path)

        self.log.info(f'Model saved: {path}')
        
        return self.model
    
    def get_metrics(self):
        
        """

        Function for generating feature model performance metrices for fitted model 

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        if not self.model:
            self.fit_model()
        
        self.pred_prob = self.model.predict(self.X_test[self.features])
        
        self.pred_class = list(map(round, self.pred_prob))

        self.confMatrix = metrics.confusion_matrix(self.y_test, self.pred_class)
        
        self.area_roc = metrics.roc_auc_score(self.y_test, self.pred_prob)

        self.log.info('AUC (test): {:.0%}'.format(self.area_roc))

        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.y_test, self.pred_class)
        
        self.precision = precision[1]

        self.log.info('Precision (test): {:.0%}'.format(self.precision))
        
        self.recall = recall[1]

        self.log.info('Recall (test): {:.0%}'.format(self.recall))
        
        self.fscore = fscore[1]

        self.log.info('F_score (test): {:.0%}'.format(self.fscore))
        
        self.precision_curve, self.recall_curve, _ = metrics.precision_recall_curve(self.y_test, self.pred_prob)
        
        self.trP, self.trN, _ = metrics.roc_curve(self.y_test, self.pred_prob)
        
        self.log.info('Roc_curve values created')

        self.df_features = pd.DataFrame({
            'feature' : self.model.pvalues.drop('Intercept').index, 
             'importance' : abs(self.model.get_margeff().margeff)
        })
        
        self.df_features = self.df_features.sort_values(by='importance', ascending=False)

        self.metrics = True


    def model_results(self):

        """

        Function fits the model, generates future importance and calculates all the relevant 
        metrices for model evalualtion

        Parameters
        ---------- 
        
        None
        
        Returns
        -------
        
        model : object
            xgb model object
        
        """
        
        if not self.model:
            self.fit_model()
        
        if not self.metrics:
            self.get_metrics()

        return self.model
        
    def plot_comf_matrix(self, cm, ax=None):
        
        """
        
        Plot confusion matrix
        
        Parameters
        ----------
        
        cm : matrix
            Confusion matrix
        
        ax : matplotlib ax
        
        Returns
        -------
        
        Matplotlib plot
        
        """
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
            
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 12}, ax=ax)
        
        ax.set_xlabel('Predicted label', fontsize=12)
        
        ax.set_ylabel('True label', fontsize=12)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_frame_on(False)
        
        ax.set_title('Confusion Matrix', fontsize=12)
        
    def plot_roc_curve(self, trP, trN, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
            
        ax.plot(trP, trN, color='orange')
        
        ax.plot([1,0], [1,0], "--", color='darkblue')
        
        ax.set_frame_on(False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        
        ax.set_ylabel('True Positive Rate', fontsize=12)
        
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12)
        
        ax.text(0.3, 0.7, 'AUC = {:.3f}'.format(self.area_roc), fontsize=12)
        
        # self.format_grid(ax)
        
    def plot_rp_curve(self, rcl, prc, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
            
        ax.plot(rcl, prc, color='orange')
        
        ax.set_frame_on(False)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_xlabel('recall', fontsize=12)
        
        ax.set_ylabel('precision', fontsize=12)
        
        ax.set_title('Precision recall Curve', fontsize=12)
        
        # self.format_grid(ax)
        
    def plot_prediction(self, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
        
        palette = {0:'deeppink', 1:'green'}
        
        df_pred = pd.DataFrame({'actual_class': self.y_test, 'pred_class': self.pred_class, 'pred_prob': self.pred_prob})
        
        sns.kdeplot(data=df_pred, x='pred_prob', hue='pred_class', fill=True, legend=False, palette=palette, ax=ax)
        
        ax.set_facecolor('#192841')
        
        ax.set_xlabel('Probability of event', fontsize=12, color='white')
        
        ax.set_ylabel('Probability Density', fontsize=12)
        
        ax.set_title('Prediction distribution', fontsize=12)
        
        ax.tick_params(axis='both', which='major')
        
        ax.grid(False)

    def plot_feature_imp(self, ax=None):
        
        if not ax:
            fig,ax = plt.subplots(figsize=(5,4))
        
        sns.barplot(x='feature', y='importance', data=self.df_features, palette='RdBu_r', ax=ax) 
        
        ax.tick_params(axis='x', rotation=90)
        
        ax.set_frame_on(False)
        
        ax.set_xlabel('Feature', fontsize=12)
        
        ax.set_ylabel('Feature importance', fontsize=12)
        
        ax.set_title('Features importance', fontsize=12)
        
    def make_plots(self):

        """

        Generate plots for evaluating the performance of the model.

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        Matplotlib plot
        
        """
        
        if not self.metrics:
            self.get_metrics()


        cm = self.confMatrix

        area_roc = self.area_roc


        trP = self.trP

        trN = self.trN


        fig = plt.figure(figsize=(20,10), constrained_layout=True)

        spec2 = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

        fig_ax1 = fig.add_subplot(spec2[0,0])
        fig_ax2 = fig.add_subplot(spec2[0,1])
        fig_ax3 = fig.add_subplot(spec2[0,2])
        fig_ax4 = fig.add_subplot(spec2[0,3])
        fig_ax5 = fig.add_subplot(spec2[1,:])

        # confusion matrix
        self.plot_comf_matrix(cm, ax=fig_ax1)

        self.plot_roc_curve(trP, trN, ax=fig_ax2)

        self.plot_rp_curve(self.recall_curve, self.precision_curve, ax=fig_ax3)

        self.plot_prediction(ax=fig_ax4)

        self.plot_feature_imp(ax=fig_ax5)

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        plt.show()  
        
