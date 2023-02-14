
import pandas as pd 
import pandas
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
import os
import json
import joblib
import time
import copy

import shap

from datetime import datetime
from collections import defaultdict

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, space_eval
from sklearn import metrics
from scipy.interpolate import make_interp_spline

from ml_processor.configuration import config
from ml_processor.jsonSerializer import NpEncoder

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')


# <<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class model_training:
    
    """

    Performing most machine learning tasks including hyperparameter tuning, model fitting and diagnostic checks.
    
    Parameters
    ----------
        
    df : pandas.Dataframe 
        Dataset with features and labels.

    features : list or array-like 
        Variable names (features) for fitting the model.

    target : string (default = "target")
        Name of column with labels (dependent variable).

    params_prop : float (default = 0.25) 
        Proportion of data set to use for hyperparameter tuning.

    test_size : float (default = 0.33) 
        Proportion of data to use as the test set.

    hyperparams : dictionary (default = None) 
        Predefined hyperparameters and their values.Specified if hyperparameter tunning is not necessary.
    
    method : string (default = "hyperopt", options = ["hyperopt", "random_search", "grid_search"])
        Hyperparameter optimization method

    classifier : string (default = "xgboost", options = ["xgboost", "random_forest", "lightgbm"])
        Classification algorithm to optimize

    eval_metric : string (default = "recall")
        Performance metric to optimise.

    search_space : dic (default = None)
        Range of values for the different parameters to tune.
    
    """

    
    def __init__(self, df, features, target="target", params_prop=0.25, test_size=0.33, hyperparams=None, 
           method="hyperopt", classifier="xgboost", eval_metric="recall", search_space=None
           ):
        
        self.data = df
        
        self.features = features
        
        self.target = target
        
        self.hyperparams = hyperparams

        self.method = method
        
        self.classifier = classifier
        
        self.eval_metric = eval_metric
        
        self.search_space = search_space
        
        self.cwd = os.getcwd()
        
        self.log = config.get_logger(file=f"{self.classifier}.log", folder=f"{self.classifier}_model_artefacts")
        
        if not os.path.isdir(f'{self.classifier}_model_artefacts'):
            os.mkdir(f'{self.classifier}_model_artefacts')

        self.path = os.path.join(f'{self.classifier}_model_artefacts', f'{self.classifier}_Model_results.json')
        
        try:
            with open(self.path) as file:
                self.model_artefacts = json.load(file)
        except:
            self.model_artefacts = {}

        self.model_name = f'{self.classifier}_' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
        
        self.log.info(f'Model job name: {self.model_name}')
        
        self.model_artefacts[self.model_name] = {}
        
        self.params_prop = params_prop
        self.model_artefacts[self.model_name]['tunning_data_size'] = self.params_prop
        
        self.test_size = test_size
        self.model_artefacts[self.model_name]['test_data_size'] = self.test_size

        self.X_train = pd.DataFrame()

        self.df_features = pd.DataFrame()


    def fit_model(self):
        
        """

        Fit model 
        
        Parameters
        ---------- 
        
        None
        
        Returns
        -------
        
        model : object
            Model object
        
        """
        
        if self.X_train.empty:
            self._split_data()
        
        if not self.hyperparams:
            self._parameter_tunning()

        best_params = self.hyperparams
        
        self.log.info('Model fitting initialized...')

        if self.classifier == "xgboost":
            model = xgb.XGBClassifier(**best_params)
        elif self.classifier == "random_forest":
            model = RandomForestClassifier(**best_params)
        elif self.classifier == "lightgbm":
            model = LGBMClassifier(**best_params)
        
        self.log.info('Model fitting started...')

        model.fit(self.X_train[self.features], self.y_train)
        
        # end = int(time.time())
        
        self.log.info('Model fitting completed')
        
        # self.log.info(f'Runtime for fitting the model : {int(end-start)} seconds')
        
        self.model = model

        self._get_metrics()

        # saving all model artefacts
        json_object = json.dumps(self.model_artefacts, cls=NpEncoder, indent=4)
        with open(self.path, 'w') as outfile:
            outfile.write(json_object)
        self.log.info(f'Model artifacts saved @:{self.path}')

        path = os.path.join(os.getcwd(), f'{self.classifier}_model_artefacts', str(self.model_name) + '.sav')

        joblib.dump(model, path)

        self.log.info(f'Model saved @:{path}')

    def model_results():

        """

        Get model results from fitted model.

        Parameters
        ---------- 
        
        None
        
        Returns
        -------
        
        object:
            Model object.
        
        """

        return self.model


    def model_evaluation(self, model = None, features = None, target = None, train_set = pd.DataFrame(), 
        validation_set = pd.DataFrame(), test_set = pd.DataFrame(), main_set = "test", loc_path = None
        ):

        """

        Generate model diagnostics plots

        Parameters
        ----------

        model : object
            Model to evaluate

        features : list or array-like (default=None)
            Features used for fitting the model

        target : string (default=None)
            COlumn with labels

        train_set : pandas.Dataframe (default=pandas.DataFrame())
            Training set to evaluate the model on. If not set, the model is evlauted on the remaining sets (validation_set, test_set). 
            Atleast one of the sets (training_set, validation_set, test_set) has to be provided. 

        validation_set : pandas.Dataframe (default=pandas.DataFrame())
            Validation set to evaluate the model on. If not set, the model is evlauted on the remaining sets (train_set, test_set). 
            Atleast one of the sets (training_set, validation_set, test_set) has to be provided. 

        test_set : pandas.Dataframe (default=pandas.DataFrame())
            Testing set to evaluate the model on. If not set, the model is evlauted on the remaining sets (train_set, validation_set). 
            Atleast one of the sets (training_set, validation_set, test_set) has to be provided.

        main_set : string (default="test", options=["test", "validation", "train"])
            Dataset to consider as the actual testing set for the model. Some model performance metrices are ony evaluated on the main set.
            The main set must be part of the data sets (training_set, validation_set, test_set) provided.

        loc_path : string (default=None):
            Directory (folder) to save the plots.

        """

        if not model:
            model = self.model 
        
        if  not features:
            features = self.features

        if not target:
            target = self.target
        
        if train_set.empty:
            train_set = pd.concat([self.X_train, self.y_train], axis=1)
        
        if test_set.empty:
            test_set = pd.concat([self.X_test, self.y_test], axis=1)
        
        if not loc_path:
            loc_path = os.path.join(f"{self.classifier}_model_artefacts", "model_performance_plot.png")

        model_perf = plot_model_perf(
            model = model,
            features = features,
            target = target,
            train_set = train_set,
            validation_set = validation_set,
            test_set = test_set,
            main_set = main_set,
            loc_path = loc_path,
            )

        return model_perf.make_plots()


    def shap_summary_plot(self, model=None, eval_set=pd.DataFrame(), save_plot=True, save_path=None, plot_size=(15,20) ):

        """

        Generate shap summary plot for fitted model

        model : object 
            Fitted model whose shap summary plot is to be generated

        eval_set : pandas.DataFrame (default=pandas.DataFrame())
            Data set to use for model evaluation

        save_plot : boolean (default=True)  
            Whether to save plot or not

        save_path : string (default = None)
            Path to save plot to.

        plot_size : "auto", float, (float, float) (default), or None.
            What size to make the plot.

        """

        if not model:
            model = self.model

        if eval_set.empty:
            eval_set = self.X_train

        if save_plot:
            if not save_path:
                save_path = os.path.join(f"{self.classifier}_model_artefacts", "shap_summary_plot.png")

        model_interp_shap(model=model, eval_set=eval_set, save_path=save_path, plot_size=plot_size )


    def _parameter_tunning(self):
        
        """

        Hyperparameter tuning using hyperopt method.

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """

        self._split_tunning_data()
        

        self.hyperparams = hyper_parameter_tunning(
            features = self.X_train_param[self.features], 
            labels = self.y_train_param, 
            method = self.method,
            classifier = self.classifier,
            eval_metric = self.eval_metric,
            search_space = self.search_space,
            )
        
        self.log.info('Hyperparameter tuning completed')
        
        # self.log.info(f'Runtime for Hyperparameter tuning : {int(end-start)} seconds')

        # self.log.info(f'Best parameters: {self.hyperparams}')
        
        self.model_artefacts[self.model_name]['best_params'] = self.hyperparams



    def _get_metrics(self):
        
        """

        Function for generating feature model performance metrices for fitted model 

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        if self.df_features.empty:
            self._feature_imp()

        metrices = get_metrics(self.model, self.X_test, self.y_test)

        predictions = get_predictions(self.model, self.X_test, self.y_test)

        recall_precision_arrays = get_prediction_arrays(self.model, self.X_test, self.y_test)
        
        self.pred_class = predictions["pred_class"]
        self.log.info('Predicted labels generated (test)')
        

        self.pred_prob = predictions["pred_prob"]
        self.log.info('Predicted probabilities generated (test)')

        self.confMatrix = metrices["confusion_matrix"]
        self.log.info('Confusion matrix generated (test)')

        self.area_roc = metrices["area_roc"]
        self.model_artefacts[self.model_name]['area_roc'] = float(self.area_roc)
        self.log.info(f"AUC (test): {self.area_roc:.0%}")

        metrices_ = ["precision", "recall", "fscore", "accuracy", "false_positive_rate"]
        for metric in metrices_:
            metric_value = float(metrices[metric])
            self.model_artefacts[self.model_name][metric] = metric_value
            self.log.info(f"{metric} (test): {metric_value:.0%}")

        self.precision_curve, self.recall_curve,  = recall_precision_arrays["precision_curve"], recall_precision_arrays["recall_curve"] 
        self.log.info('Precision and Recall values for the precision recall curve created')


        self.trP, self.trN = recall_precision_arrays["trP"], recall_precision_arrays["trN"]
        self.log.info('True positive and negativevalues for the ROC curve created')


    def _feature_imp(self):
        
        """

        Function for generating feature importance for fitted model 
        
        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """

        try:

            feature_importance = get_variableImportance(self.model, self.features) 
            
            self.log.info('Dataframe with feature importance generated')
            
            self.df_features = pd.DataFrame(feature_importance)
            
            self.model_artefacts[self.model_name]['feature_importance'] = {
                'feature' : list(self.features), 
                'importance' : list(self.model.feature_importances_)
            }

        except:
            self.log.warning(f"!!! Model object not found, fit model first and proceed")
            

    def _split_data(self): 
        
        """

        Split data into training and test sets

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(data=self.data, target=self.target, test_size=self.test_size)
        
        self.log.info('Splitting data into training and testing sets completed')
        
        self.log.info(f'Training data set:{self.X_train.shape[0]} rows')
        
        self.log.info(f'Testing data set:{self.X_test.shape[0]} rows')


    def _reduce_data(self):
        
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


    def _split_tunning_data(self):
        
        """

        Split data into training and test sets

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        None
        
        """
        
        tunning_data = self._reduce_data()

        self.X_train_param, self.X_test_param, self.y_train_param, self.y_test_param = split_data(data=tunning_data, target=self.target, test_size=self.test_size)
        
        self.log.info('Splitting hyperparameter tuning data into training and testing sets completed')
        
        self.log.info(f'Hyperparameter tuning training data set:{self.X_train_param.shape[0]} rows')
        
        self.log.info(f'Hyperparameter tuning testing data set:{self.X_test_param.shape[0]} rows')



# <<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class plot_model_perf:

    """

    Generate model diagnostics plots

    Parameters
    ----------

    model : object
        Model to evaluate

    features : list or array-like (default=None)
        Features used for fitting the model

    target : string (default="target")
        COlumn with labels

    train_set : pandas.Dataframe (default=pandas.DataFrame())
        Training set to evaluate the model on. If not set, the model is evlauted on the remaining sets (validation_set, test_set). 
        Atleast one of the sets (training_set, validation_set, test_set) has to be provided. 

    validation_set : pandas.Dataframe (default=pandas.DataFrame())
        Validation set to evaluate the model on. If not set, the model is evlauted on the remaining sets (train_set, test_set). 
        Atleast one of the sets (training_set, validation_set, test_set) has to be provided. 

    test_set : pandas.Dataframe (default=pandas.DataFrame())
        Testing set to evaluate the model on. If not set, the model is evlauted on the remaining sets (train_set, validation_set). 
        Atleast one of the sets (training_set, validation_set, test_set) has to be provided.

    main_set : string (default="test", options=["test", "validation", "train"])
        Dataset to consider as the actual testing set for the model. Some model performance metrices are ony evaluated on the main set.
        The main set must be part of the data sets (training_set, validation_set, test_set) provided.

    loc_path : string (default=None):
        Directory (folder) to save the plots.

    """

    def __init__(self, model, features, target="target", train_set=pd.DataFrame(), 
        validation_set=pd.DataFrame(), test_set=pd.DataFrame(), main_set="test", loc_path=None
        ):


        self.model = model 
        self.features = features
        self.target = target
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.main_set = main_set

        if loc_path:
            self.loc_path = loc_path
        else:
            self.loc_path = os.path.join(os.getcwd(), "model_performance_plot.png")

        self.columnNames = {}

        self.eval_sets = ["train", "validation", "test"]
        self.eval_sets_dic = {}

        # self.main_data_set  = eval(f"self.{self.main_set}_set")

        for eval_set in self.eval_sets:

            data  = eval(f"self.{eval_set}_set")

            if isinstance(data, pandas.core.frame.DataFrame) and not data.empty:

                columnName = f"{eval_set}(size={len(data):,})"
                self.columnNames[eval_set] = columnName

                self.eval_sets_dic[eval_set] = {}

                predictions = get_predictions(self.model, data[self.features], data[self.target])
                self.eval_sets_dic[eval_set]["predictions"] = predictions

                _metrices = get_metrics(model, data[self.features], data[self.target])
                self.eval_sets_dic[eval_set]["_metrices"] = _metrices

                pred_array = get_prediction_arrays(model, data[self.features], data[self.target])
                self.eval_sets_dic[eval_set]["pred_array"] = pred_array

        self.df_features = get_variableImportance(self.model, self.features, returnType="df")

    def get_metric_df(self):

        """

        Generate dataframe with model performance metrics

        Parameters
        ----------
        
        None
        
        Returns
        -------
        
        df_metrices : pandas.DataFrame
            Metrics data
        
        """

        df_metrices = pd.DataFrame()

        for i, eval_set in enumerate(self.eval_sets_dic):
            metric = self.eval_sets_dic[eval_set]["_metrices"].copy()
            del metric["confusion_matrix"]
            metric = pd.Series(metric)
            metric.name = eval_set
            df_metrices = pd.concat([df_metrices, metric], axis=1)
        
        df_metrices = df_metrices.rename(columns = self.columnNames)

        return df_metrices

    def make_plots(self, display_metrics=True):

        """

        Generate model performance plots

        Parameters
        ----------
        
        display_metrics : boolean (default=True)
            Whether to display metrics table or not.
        
        Returns
        -------
        
        matplotlib plot 
        
        """

        if display_metrics:
            display(self.get_metric_df())

        sns.set_style('whitegrid')
        colors = {"train":"black", "validation":"orange", "test":"deepskyblue"}

        if self.df_features.empty:
            width = 6
            rows = 1
        else:
            width = 12
            rows = 2
        
        fig = plt.figure(figsize=(25, width), constrained_layout=True)

        spec2 = gridspec.GridSpec(ncols=5, nrows=rows, figure=fig)
        
        fig_ax1 = fig.add_subplot(spec2[0,0])
        fig_ax2 = fig.add_subplot(spec2[0,1])
        fig_ax3 = fig.add_subplot(spec2[0,2])
        fig_ax4 = fig.add_subplot(spec2[0,3])
        fig_ax5 = fig.add_subplot(spec2[0,4])

        # confusion matrix
        confMatrix = self.eval_sets_dic[self.main_set]["_metrices"].get("confusion_matrix")
        _plot_comf_matrix(confMatrix, ax=fig_ax1, dataset=self.main_set)

        # predictions
        pred_prob = self.eval_sets_dic[self.main_set]["predictions"].get("pred_prob")
        pred_class = self.eval_sets_dic[self.main_set]["predictions"].get("pred_class")
        y_test = eval(f"self.{self.main_set}_set")[self.target]
        _plot_prediction(y_test, pred_class, pred_prob, ax=fig_ax4, dataset=self.main_set)

        for i, eval_set in enumerate(self.eval_sets_dic):
            trP = self.eval_sets_dic[eval_set]["pred_array"].get("trP")
            trN = self.eval_sets_dic[eval_set]["pred_array"].get("trN")

            area_roc = self.eval_sets_dic[eval_set]["_metrices"].get("area_roc")

            recall_curve = self.eval_sets_dic[eval_set]["pred_array"].get("recall_curve")
            precision_curve = self.eval_sets_dic[eval_set]["pred_array"].get("precision_curve")
            # AP = self.eval_sets_dic[eval_set]["pred_array"].get("AP")

            color = colors.get(eval_set)

            _plot_roc_curve(trP, trN, area_roc, ax=fig_ax2, color=color, label=eval_set)
            _plot_rp_curve(recall_curve, precision_curve, ax=fig_ax3, color=color, label=eval_set)

        _plot_lift_curve(y_test, pred_prob, ax=fig_ax5, dataset=self.main_set)

        if not self.df_features.empty:
            
            fig_ax6 = fig.add_subplot(spec2[1,:])
            _plot_feature_imp(self.df_features, ax=fig_ax6)

        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        plt.savefig(self.loc_path)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------

def split_data(data, target="target", test_size=0.33): 
    
    """

    Split data into training and test sets

    Parameters
    ----------
    
    df : pandas.Dataframe 
        Full dataset to split into training and testing set. Includes both features and labels.

    target : string (default = "target")
        Name of column with labels (dependent variable).

    test_size : float (default=0.33) 
        Proportion of data to use as the test set.

    Returns
    -------
    
    X_train, X_test, y_train, y_test : pandas.DataFrames
        Where ;
            X_train : training features data
            X_test  : testing features data
            y_train : training labels data
            y_test  : testing labels data
    
    """
    
    X = data.drop(target, axis=1)
    
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test



def model_interp_shap(model, eval_set, save_path=None, plot_size=(10,10)):

    """

    Generate shap summary plot for fitted model

    model : object 
        Fitted model whose shap summary plot is to be generated

    eval_set : pandas.DataFrame (default=pandas.DataFrame())
        Data set to use for model evaluation

    save_path : string (default = None)
        Path to save plot to.

    plot_size : "auto", float, (float, float) (default), or None.
        What size to make the plot.

    """

    features = eval_set.columns
    num_cols = len(features)
    
    if isinstance(model, xgb.sklearn.XGBClassifier):
        
        explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(eval_set)

        shap.summary_plot(shap_values, eval_set, max_display=num_cols, show=False, plot_size=plot_size)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',dpi=100)

        plt.show()
    
    elif isinstance(model, LGBMClassifier) or isinstance(model, RandomForestClassifier):
        
        explainer = shap.TreeExplainer(model)
        
        shap_values_ = explainer(eval_set)
        
        shap_values = copy.deepcopy(shap_values_)
        
        shap_values.values = shap_values.values[:,:,1]
        
        shap_values.base_values = shap_values.base_values[:,1]
        
        shap_plot = shap.plots.beeswarm(shap_values, max_display=num_cols, show=False, plot_size=plot_size)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',dpi=100)

        plt.show()



def hyper_parameter_tunning(features, labels, method="hyperopt", classifier="xgboost", 
        eval_metric="recall", search_space=None):
    
    """

    Hyperparameter tuning for specified algorithm

    Parameters
    ----------
    
    features : pandas.Dataframe 
        Dataset with features

    labels : pandas.Dataframe 
        Dataset with features

    # method : string (default = "hyperopt", options = ["hyperopt", "random_search", "grid_search"])
    #     Hyperparameter optimization method --- in future versions

    classifier : string (default = "xgboost", options = ["xgboost", "random_forest", "lightgbm"])
        Classification algorithm to optimize

    eval_metric : string (default = "recall")
        Performance metric to optimise.

    search_space : dic (default = None)
        Range of values for the different parameters to tune.
    
    Returns
    -------
    
    dic :
        Hyperparameter values
    
    """

    if not search_space:
        space = search_space_generator(classifier)
    else :
        space = search_space
    
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def objective_func(params):
        
            if classifier == "xgboost":
                model = xgb.XGBClassifier(**params)
            elif classifier == "random_forest":
                model = RandomForestClassifier(**params)
            elif classifier == "lightgbm":
                model = LGBMClassifier(**params)

            scores = cross_val_score(
                model,
                features, 
                labels,
                cv=kfold, 
                scoring=eval_metric, 
                n_jobs=-1
            )

            best_score = np.mean(scores)

            loss = - best_score

            return {'loss':loss, 'params':params, 'status':STATUS_OK}
    
    print('Trials initialized...')
    
    trials = Trials()

    param_search = fmin(fn = objective_func, space=space, algo=tpe.suggest, max_evals=48, trials=trials)

    hyperparams = space_eval(space, param_search)

    return hyperparams



def get_predictions(model, X_test, y_test, threshold=0.5):

    """

    Generate probability and class predictions from fitted model

    Parameters
    ----------

    model : object
        Model to use for making predictons

    X_test : pandas.DataFrame
        Data set to make predictions on; should include only features used in fitting the model.

    y_test : list or array-like
        Test labels

    threshold : float (default=0.5)
        Probability cut-off value for class allocation 
        e.g threshold=0.5 means all samples where the predicted probability is atleast 0.5 are allocated the positive label.

    Returns
    -------

    predictions : dic
        Predicted probabilities and classes 
            • pred_prob - predicted probabilities
            • pred_class - predicted classes

    """

    if isinstance(model, xgb.core.Booster):
        
        data = xgb.DMatrix(X_test, y_test)
        
        pred_prob = model.predict(data)
    
    elif not hasattr(model, "predict_proba") and hasattr(model, "predict"):
        
        pred_prob = model.predict(X_test)
    
    elif hasattr(model, "predict_proba") and hasattr(model, "predict"):
        
        pred_prob = model.predict_proba(X_test)[:,1]

    pred_class = pred_prob>threshold
    
    pred_class = pred_class.astype(int)

    predictions = {"pred_prob": pred_prob, "pred_class": pred_class}

    return predictions



def get_metrics(model, X_test, y_test):

    """

    Generate model performance metrics

    Parameters
    ----------

    model : object
        Model for which to obtain performance metrics

    X_test : pandas.DataFrame
        Data set to make predictions on; should include only features used in fitting the model.

    y_test : list or array-like
        Test labels


    Returns
    -------

    _metrics : dic
        Model performance metrices
            • area_roc - Area under the ROC Curve.
            • accuracy - accuracy
            • precision - precision
            • recall - Recall/sensitivity
            • fscore - F1-score
            • false_positive_rate - False positive rate
            • confusion_matrix - confusion matrix 


    """

    predictions = get_predictions(model, X_test, y_test)
    pred_prob = predictions.get("pred_prob")
    pred_class = predictions.get("pred_class")

    area_roc = metrics.roc_auc_score(y_test, pred_prob)
    
    accuracy = metrics.accuracy_score(y_test, pred_class)
    
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, pred_class)
    
    confMatrix = metrics.confusion_matrix(y_test, pred_class)
    
    false_positive_rate = confMatrix[0,1]/confMatrix[0,:].sum()

    _metrics = {
        "area_roc": area_roc,
        "accuracy": accuracy,
        "precision": precision[1],
        "recall": recall[1],
        "fscore": fscore[1],
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": confMatrix
    }

    return _metrics



def get_prediction_arrays(model, X_test, y_test):

    """

    Genertae true positive-true negative, precision-recall pairs for different probability thresholds.


    Parameters
    ----------

    model : object
        Model for which to obtain performance metrics

    X_test : pandas.DataFrame
        Data set to make predictions on; should include only features used in fitting the model.

    y_test : list or array-like
        Test labels


    Returns
    -------

    pred_array : dic
        True positive, true negative, precision, recall arrays
            • trP - True positive array
            • trN - True negative  array
            • precision_curve - precision  array
            • recall_curve - recall  array

    """

    predictions = get_predictions(model, X_test, y_test)
    pred_prob = predictions.get("pred_prob")
    pred_class = predictions.get("pred_class")
    
    trP, trN, _ = metrics.roc_curve(y_test, pred_prob)
    
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_test, pred_prob)

    pred_array = {
        "trP": trP,
        "trN": trN,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }
    
    return pred_array



def get_variableImportance(model, features, returnType="dic"):

    """
    Generate variable importance for each variable in fitting a model


    Parameters
    ----------

    model : object
        Model for which to obtain variable importance

    features : list or array-like
        Features used fro fitting model; should be in the same order as in the model fitting data set.

    returnType : string (defaul="dic", options=["dic", "df"])
        Returns dictionary if set to "dic" and pandas.DataFrame if set to "df"

    Returns
    -------

    featureImportance/df_featureImp : dic/pandas.DataFrame
        Variable importance values for each variable

    """

    if isinstance(model, xgb.core.Booster):
        feature_scores = model.get_score(importance_type='gain')
        featureImportance = {"feature" : feature_scores.keys(), 'importance' : feature_scores.values()}
    
    elif hasattr(model, "feature_importances_"):
        featureImportance = {"feature" : features, 'importance' : model.feature_importances_}
    
    else:
        featureImportance = {}

    if returnType == "dic":
        return featureImportance
    elif returnType == "df":
        df_featureImp = pd.DataFrame(featureImportance)

        if not df_featureImp.empty:
            df_featureImp = df_featureImp.sort_values("importance", ascending=False)
        
        return df_featureImp



def search_space_generator(classifier="xgboost"):
    """

    Generates search space for hyperparameter tuning for specified algorithm


    Parameters
    ----------
    
    classifier : string (default = "xgboost)
    
    Returns
    -------
    
    dic:
        Range of hyperparameters to search from
    
        """
    space = {
        "xgboost" : {
            'n_estimators': hp.choice('n_estimators', range(100,1500,100)),
            'learning_rate': hp.choice('learning_rate', np.arange(0.01, 0.5, 0.01)),
            'max_depth': hp.choice('max_depth', range(2, 15, 1)),
            'subsample': hp.choice('subsample', np.arange(0.1, 1.1, 0.1)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.1, 1.1, 0.1)),
            'gamma': hp.choice('gamma', np.arange(0.1, 0.6, 0.1)),
            'reg_alpha': hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
            'reg_lambda': hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100]),
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'importance_type': 'gain',
            'random_state': 801,
            'seed': 0,
            'verbosity': 1,
            'n_jobs': -1,
        },

        "random_forest" : {
            'n_estimators': hp.choice('n_estimators', range(100,1500,100)),
            'max_depth': hp.choice('max_depth', range(2, 15, 1)),
            'criterion': 'gini',
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
            'max_features' : hp.choice('max_features', ['auto', 'sqrt']),
            'bootstrap': True,
            # 'importance_type': 'gain',
            'random_state': 801,
            'verbose' : 0,
            'n_jobs' : -1,
            
        },

        "lightgbm" : {
            'num_leaves' : hp.choice('num_leaves', range(20, 220, 20)),
            'max_depth': hp.choice('max_depth', range(2, 15, 1)),
            'learning_rate' : hp.choice('learning_rate', np.arange(0.01, 0.5, 0.01)),
            'n_estimators': hp.choice('n_estimators', range(100,1500,100)),
            'subsample' : hp.choice('subsample', np.arange(0.6, 1, 0.05)),
            'min_child_samples' : hp.choice('min_child_samples', range(5,50,10)),
            'boosting_type' : 'gbdt',
            'objective': 'binary',
            'importance_type': 'gain',
            'random_state': 801,
            'seed': 0,
            'n_jobs': -1,
        }
    }

    return space.get(classifier)

# ------------------------------------------->>>>>>>>>>>>>>>>>>


def _plot_comf_matrix(cm, ax, dataset="test"):
        
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 12}, ax=ax)
    
    ax.set_xlabel('Predicted label', fontsize=14)
    
    ax.set_ylabel('True label', fontsize=14)

    ax.set_title(f"Confusion matrix ({dataset})", fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.set_frame_on(False)


def _plot_roc_curve(trP, trN, area_roc, ax, color="orange", label="test"):
        
    ax.plot(trP, trN, color=color, label= f"{label} (AUC ≈ {area_roc:.3f})")
    
    ax.plot([1,0], [1,0], "--", color='darkblue')
    
    ax.set_frame_on(False)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.set_xlabel('False Positive Rate', fontsize=14)
    
    ax.set_ylabel('True Positive Rate', fontsize=14)
    
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    
    # ax.text(0.3, 0.7, 'AUC = {:.3f}'.format(area_roc), fontsize=12)

    ax.legend(frameon=False)


def _plot_rp_curve(rcl, prc, ax, color="orange", label="test"):
        
    ax.plot(rcl, prc, color=color, label= f"{label}")
    
    ax.set_frame_on(False)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.set_xlabel('recall', fontsize=14)
    
    ax.set_ylabel('precision', fontsize=14)
    
    ax.set_title('Precision recall Curve', fontsize=16)

    ax.legend(frameon=False)


def _plot_prediction(y_test, pred_class, pred_prob, ax, dataset="test"):
    
    palette = {0:'deeppink', 1:'green'}
    
    df_pred = pd.DataFrame({'actual_class': y_test, 'pred_class': pred_class, 'pred_prob': pred_prob})
    
    sns.kdeplot(data=df_pred, x='pred_prob', hue='pred_class', fill=True, legend=False, palette=palette, ax=ax)
    
    ax.set_facecolor('#274472')
    
    ax.set_xlabel('Probability of event', fontsize=14, color='white')
    
    ax.set_ylabel('Probability Density', fontsize=14)
    
    ax.set_title(f'Probability distribution ({dataset})', fontsize=16)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(False)



def _plot_feature_imp(df_features, ax):
    
    sns.barplot(x='feature', y='importance', data=df_features, palette='RdBu_r', ax=ax) 
    
    ax.tick_params(axis='x', rotation=90)
    
    ax.set_frame_on(False)
    
    ax.set_xlabel('Feature', fontsize=14)
    
    ax.set_ylabel('Feature importance', fontsize=14)
    
    ax.set_title('Features importance', fontsize=16)



def _plot_lift_curve(y_test, pred_prob, ax=None, output="plot", dataset="test"):
    test_set = pd.DataFrame({"target" : y_test, "probability" : pred_prob})
    
    test_set = test_set.sort_values(by="probability", ascending=False)
    
    test_set["decile"] = np.linspace(1, 11, len(test_set), False, dtype=int)

    df_gain = (
        test_set
        .groupby("decile")
        .agg(
            count = ("target", "count"),
            events = ("target", sum) 
        )
        .assign(pct_events = lambda X: X["events"]/X["events"].sum()*100)
        .assign(gain = lambda X: X["pct_events"].cumsum())
        .reset_index()
        .assign(decile_pct = lambda X: X["decile"]*10)
        .assign(lift = lambda X: X["gain"]/X["decile_pct"])
    )

    if output != "plot":
        return df_gain
    else:
        if not ax:
            fig, ax = plt.subplots(figsize=(8,6))

        x = np.array(df_gain.decile_pct)
        y = np.array(df_gain.lift)

        X_Y_Spline = make_interp_spline(x, y)

        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)

        ax.plot(X_, Y_,color='orange', lw=5, label="Lift curve")
        ax.plot(X_, [1 for x in Y_], ls="--", color='black', lw=3, label="Baseline")

        ax.set_xticklabels([f"{x:.0f}%" for x in ax.get_xticks()])

        ax.legend(loc="center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2)

        ax.set_xlabel("Percentage of sample", labelpad=10, fontsize=14)
        ax.set_ylabel("Lift", labelpad=10, fontsize=14)

        ax.set_title(f"Lift curve ({dataset})", fontsize=16)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)



