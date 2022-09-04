import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

import os
import json
from datetime import datetime


import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, space_eval
from sklearn import metrics



from ml_processor.configuration import config

# from configuration import config 
from ml_processor.jsonSerializer import NpEncoder

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')



class xgb_training:
    
    """
    Class for performing machine learning tasks including hyperparameter tuning and xgb model fitting
    
    Attributes:
        
        df (pandas dataframe) dataset with features and labels
        
        features(list) features for fitting the model
        
        target (string) name of column with labels (dependent variable)
        
        params_prop (float) proportion of data set to use for hyperparameter tuning
        
        test_size (float) proportion of data to use as the test set
        
        hyperparams (dictionary) Predefined hyperparameters and their values. 
            Specified if hyperparameter tunning is not necessary
    """

    
    def __init__(self, df, features, target, params_prop=0.25, test_size=0.33, hyperparams=None):
        
        self.data = df
        
        self.features = features
        
        self.target = target
        
        self.hyperparams = hyperparams
        
        self.cwd = os.getcwd()
        
        self.log = config.get_logger(file='xgbModelLogs.log')
        
        self.path = os.path.join(self.cwd, 'xgb_model_results.json')
        
        try:
            with open(self.path) as file:
                self.model_artefacts = json.load(file)
        except:
            self.model_artefacts = {}

        self.model_name = 'model_' + str(int(datetime.now().timestamp()))
        
        self.log.info(f'Model job name: {self.model_name}')
        
        self.model_artefacts[self.model_name] = {}
        
        self.params_prop = params_prop
        self.model_artefacts[self.model_name]['tunning_data_size'] = self.params_prop
        
        self.test_size = test_size
        self.model_artefacts[self.model_name]['test_data_size'] = self.test_size
    
        
    def split_data(self):
        
        """
        Function for spliting data into training and test sets

        Args:
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
        Function for generating hyperparameter tuning data 

        Args:
            None
        
        """
        
        tunning_data_size = int(self.data.shape[0] * self.params_prop)
        
        self.log.info('Hyper parameter tunning data set created')
    
        self.log.info(f'Hyper parameter tunning data set:{tunning_data_size} rows')
        
        return self.data.sample(tunning_data_size)

        
    def split_tunning_data(self):
        
        """
        Function for spliting hyperparameter tuning data into training and test sets

        Args:
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
        Function for hyperparameter tuning using hyperopt method

        Args:
            None
        
        """
        
        self.split_tunning_data()
  
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
        
        self.log.info('Hyperparameter tuning successfully completed')
        
        self.hyperparams = space_eval(space, param_search)
        
        self.log.info(f'Best parameters: {self.hyperparams}')
        
        self.model_artefacts[self.model_name]['best_params'] = self.hyperparams
        
    def fit_model(self):
        
        """
        Function for fitting model 

        Args:
            None
        
        """
        
        self.split_data()
        
        if not self.hyperparams:
            self.hyper_parameter_tunning()

        best_params = self.hyperparams
        
        self.log.info('Model fitting initialized...')

        xgb_model = xgb.XGBClassifier(seed=0, **best_params)
        
        self.log.info('Model fitting started...')

        xgb_model.fit(self.X_train[self.features], self.y_train)
        
        self.log.info('Model fitting succesfully completed')
        
        self.model = xgb_model
        
    def get_feature_imp(self):
        
        """
        Function for generating feature importance for fitted model 

        Args:
            None
        
        """
        
        self.df_features = pd.DataFrame({'feature' : self.features, 'importance' : self.model.feature_importances_})

        self.df_features = self.df_features.sort_values(by='importance', ascending=False)
        
        self.log.info('Dataframe with feature importance generated')
        
        self.model_artefacts[self.model_name]['feature_importance'] = {'feature' : list(self.features), 'importance' : list(self.model.feature_importances_)}
    
    
    def get_metrics(self):
        
        """
        Function for generating feature model performance metrices for fitted model 

        Args:
            None
        
        """
        
        # function generates feature importance
        self.get_feature_imp()
        
        # predicted labels
        self.pred_class = self.model.predict(self.X_test[self.features])
        self.model_artefacts[self.model_name]['pred_class'] = list(self.pred_class)
        self.log.info('Predicted labels generated (test)')
        
        # predicted probabilities
        self.pred_prob = self.model.predict_proba(self.X_test[self.features])[:,1]
        self.model_artefacts[self.model_name]['pred_prob'] = list(self.pred_prob)
        self.log.info('Predicted probabilities generated (test)')

        # confusion matrix
        self.confMatrix = metrics.confusion_matrix(self.y_test, self.pred_class)
#         self.model_artefacts[self.model_name]['confMatrix']  = self.confMatrix
        self.log.info('Confusion matrix generated (test)')
        
        # AUC
        self.area_roc = metrics.roc_auc_score(self.y_test, self.pred_prob)
        self.model_artefacts[self.model_name]['area_roc'] = float(self.area_roc)
        self.log.info(f"Area Under the Curve (AUC) (test): {self.model_artefacts[self.model_name]['area_roc']}")

        # preciosn, recall and fscore
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.y_test, self.pred_class)
        
        self.model_artefacts[self.model_name]['precision'] = float(precision[1])
        self.log.info(f"Precision at default threshold (test): {precision[1]}")

        self.model_artefacts[self.model_name]['recall'] = float(precision[1])
        self.log.info(f"Recall at default threshold (test): {recall[1]}")

        self.model_artefacts[self.model_name]['fscore'] = fscore[1]
        self.log.info(f"F_score at default threshold (test): {recall[1]}")
        
        # precison, recall for the recall_precison curve
        self.precision_curve, self.recall_curve, _ = metrics.precision_recall_curve(self.y_test, self.pred_prob)
        
        self.model_artefacts[self.model_name]['precision_curve'] = list(self.precision_curve)
        self.log.info('Precision values for the precision recall curve created')
        
        self.model_artefacts[self.model_name]['recall_curve'] = list(self.recall_curve)
        self.log.info('Recall values for the precision recall curve created')
        
        # true positive and true negative for te ROC curve
        self.trP, self.trN, _ = metrics.roc_curve(self.y_test, self.pred_prob)
        
        self.model_artefacts[self.model_name]['trP'] = list(self.trP)
        self.log.info('True positive values for the ROC curve created')

        self.model_artefacts[self.model_name]['trN'] = list(self.trN)
        self.log.info('True negative values for the ROC curve created')
        
        # precision, recall at different thresholds
        perf_diff_thresholds = {'thresholds':[], 'precision':[], 'recall':[], 'f1_score':[] }
        
        perf_diff_thresholds['thresholds'] = np.arange(0, 1+1e-5, 0.05)
        
        for i in perf_diff_thresholds['thresholds']:
            
            precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
                self.y_test, self.model_artefacts[self.model_name]['pred_prob'] > i, average='binary')
            
            perf_diff_thresholds['precision'].append(precision)
            
            perf_diff_thresholds['recall'].append(recall)
            
            perf_diff_thresholds['f1_score'].append(f1_score)
        
#         perf_diff_thresholds =  pd.DataFrame(perf_diff_thresholds)
        
        self.log.info('Recall and precision calculation for different thresholds (test) completed')
        
        self.model_artefacts[self.model_name]['df_diff_thresholds'] = perf_diff_thresholds
    
    def model_results(self):
        """
        Function fits the model, generates future importance and calculates all the relevant metrices for model evalualtion

        Args:
            None
        
        Returns:
            object: xgb model object
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
        
        df_pred = pd.DataFrame({'actual_class': self.y_test, 'pred_class': self.model_artefacts[self.model_name]['pred_class'], 'pred_prob': self.model_artefacts[self.model_name]['pred_prob']})
        
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
        
        if not self.model:
            log.error('Fit model first using model_results method')
        else:

            cm = self.confMatrix

            area_roc = self.model_artefacts[self.model_name]['area_roc']


            trP = self.model_artefacts[self.model_name]['trP']

            trN = self.model_artefacts[self.model_name]['trN']


            fig = plt.figure(figsize=(15,10), constrained_layout=True)

            spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

            fig_ax1 = fig.add_subplot(spec2[0,0])
            fig_ax2 = fig.add_subplot(spec2[0,1])
            fig_ax3 = fig.add_subplot(spec2[0,2])
            fig_ax4 = fig.add_subplot(spec2[1,0])
            fig_ax5 = fig.add_subplot(spec2[1,1:])

            # confusion matrix
            self.plot_comf_matrix(cm, ax=fig_ax1)

            self.plot_roc_curve(trP, trN, ax=fig_ax2)

            self.plot_rp_curve(self.recall_curve, self.precision_curve, ax=fig_ax3)

            self.plot_prediction(ax=fig_ax4)

            self.plot_feature_imp(ax=fig_ax5)

            plt.subplots_adjust(hspace=0.3, wspace=0.3)

            plt.show()

