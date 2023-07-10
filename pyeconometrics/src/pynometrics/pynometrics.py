import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt 
import seaborn as sns
import logging
import statsmodels.api as sm 
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import collections as matcoll
from stargazer.stargazer import Stargazer


def log_info(loglevel=logging.INFO, file=None):

    logger = logging.getLogger(__name__)

    logger.setLevel(loglevel)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    if not logger.handlers:
        if file:

            filehandler = logging.FileHandler(file)

            filehandler.setLevel(logging.ERROR)

            filehandler.setFormatter(formatter)

            logger.addHandler(filehandler)

        else:
            streamhandler = logging.StreamHandler()

            streamhandler.setFormatter(formatter)

            logger.addHandler(streamhandler)
        
    logger.propagate = False

    return logger

class get_marginal_effects():
    """
    Generic function to calculate the odds ratio and marginal effects from a logistic regression model 
    fitted with statsmodels.api
    
    Attributes:
    logit_model (object) statsmodel ols model results
    """
    
    logger = log_info()
    
    def __init__(self,logit_model):
        
        self.logit_model = logit_model
        
        self.model = copy.deepcopy(logit_model)
        
        self.marginal_effects = self.model.get_margeff()
    
    def get_margins(self):

        self.model.params[1:] = self.marginal_effects.margeff
        
        self.model.bse[1:] = self.marginal_effects.margeff_se
        
        self.model.tvalues[1:] = self.marginal_effects.tvalues
        
        self.model.pvalues[1:] = self.marginal_effects.pvalues 
        
        self.model.conf_int()[1:][0] = self.marginal_effects.conf_int()[:,0]
        
        self.model.conf_int()[1:][1] = self.marginal_effects.conf_int()[:,1]
        
        return self.model
    
        self.logger.info("Margins generation successful")
    
    def plot_coeff_margins(self):

        df_params = (
            
            pd.DataFrame(self.logit_model.params)
            
            .reset_index()
            
            .rename(columns={"index":"var", 0:"coef"})
            
            .assign(
                lower_CI = self.logit_model.conf_int()[0].values,
                
                errors = lambda X: X['coef'] - X['lower_CI'],
                
                pvalues = self.logit_model.pvalues.values,
                
                Significant = lambda X: X['pvalues'] < 0.05 )
            
            .drop(0) 
        )
        
        self.logger.info("Coefficients dataframe generation successful")

        df_margins = (
            
            self.marginal_effects.summary_frame()
            
            .reset_index()
            
            .iloc[:,[0,1,4,5]]
            
            .rename(columns={"index":"var" ,"dy/dx":"coef" ,"Pr(>|z|)":"pvalues","Conf. Int. Low":"lower_CI"})
            
            .assign(
                errors = lambda X: X['coef'] - X['lower_CI'],
                
                Significant = lambda X: X['pvalues'] < 0.05)
        )
        
        self.logger.info("Margins dataframe generation successful")
        

        params = zip([df_params, df_margins], ['coefficients', 'margins'])
        
        fig, ax = plt.subplots(1,2, figsize = (16,6))
        
        sns.set_style("whitegrid")
        
        for i, (df, col) in enumerate(params):
            
            ax = plt.subplot(1, 2, i+1)
            
            df.plot(x = 'var', y = 'coef', ax = ax, kind = 'barh', color = 'none', fontsize = 22, 
                           ecolor = df.Significant.map({True: '#20B2AA', False: '#F08080'}),
                           capsize = 0, xerr = 'errors', legend = False)
            
            # locate the coefficients on the confidence interval
            plt.scatter(y = np.arange(df.shape[0]), x = df['coef'],  marker = 'o', s = 160, 
                       color = df.Significant.map({True: '#20B2AA', False: '#F08080'}))

            ax.axvline(x = 0, linestyle = '--', color = '#F08080', linewidth = 1)

            plt.title('95% CI for the {}'.format(col), fontsize = 16, )

            plt.tick_params(axis = 'both', which = 'major', labelsize = 10)

            plt.ylabel('')

            plt.xlabel('estimates',fontsize = 12, )

        return plt.show()  
    

    def odd_ratios(self):
        
        self.model1 = copy.deepcopy(self.logit_model)
        
        df_odds = pd.concat([self.model1.params, self.model.conf_int(),], axis=1)
        
        df_odds.columns = ['OR', '2.5%', '97.5%']
        
        return df_odds.apply(lambda x: np.exp(x))   

class mplot:
    """
    Class for plotting different OLS model iagnostic checks. 
    Reproduces the diagnostic plots in mplot package in R
    
    Attributses:
        mdl (object) representing statsmodels OLS model results
        n (int) representing the diagnostic check to plot
            1 = Residuals vs Fitted
            2 = Normal Q-Q
            3 = Scale-Location
            4 = Cook's Distance
            5 = Residuals vs Leverage
            6 = Cook's dist vs Leverage
            7 = 95% confidence intervals
    """
 
    def __init__(self, mdl):
        
        self.mdl = mdl
        
        self.fitted_values = pd.Series(self.mdl.fittedvalues)
        
        self.res_df = (
            
            pd.DataFrame(
                self.fitted_values, index = self.mdl.fittedvalues.index, columns = ['fitted_values'])
           
            .assign(
                residuals = self.mdl.resid,
                student_residuals = self.mdl.get_influence().resid_studentized_internal,
                sqrt_student_residuals = lambda X: np.sqrt(np.abs(X['student_residuals'])),
                leverage = self.mdl.get_influence().hat_matrix_diag,
                cooks_distance = self.mdl.get_influence().cooks_distance[0])
        )
        
        self.df_params = (
            
            pd.DataFrame(self.mdl.params.reset_index())
            .rename(columns={"index":"var", 0:"coef"})
            .assign(
                errors = lambda X: X['coef'] - self.mdl.conf_int().iloc[:,0].values,
                pvalues = self.mdl.pvalues.values,
                Significant = lambda X: X['pvalues'] < 0.05)
            .query("var != 'Intercept'")
        )

        
    def plot(self, n, figsize=(12,5)):
        """
        Function to generate plot for chosen diagnostic check for linear regression
        
        Args:
            None
        
        Returns
            Matplotlib plot: Plot of chosen diagnostic check
        """
        
        sns.set_style('whitegrid')
        
        params = {
            "figure.figsize":figsize,
            "text.color":"#162871",
            "axes.titlesize":16,
            "axes.labelsize":14,
            "axes.labelcolor": "#162871",
            "axes.edgecolor": "#162871",
            "xtick.color": "#162871",
            "ytick.color": "#162871",
            "xtick.labelsize":10,
            "ytick.labelsize":10,
            "legend.fontsize":12, 
            "axes.grid.axis":"y",
            "axes.spines.left":False,
            "axes.spines.top":False,
            "axes.spines.right":False,

        }
        
        plt.rcParams.update(params)
        
        from matplotlib.figure import Figure
        
        fig = Figure()
        ax = fig.subplots()
        
#         fig, ax = plt.subplots()
        
        if n == 1:
            
            # rows with top residuals 
            top3 = abs(self.res_df['residuals']).sort_values(ascending = False)[:3]
            
            X, Y = self.res_df['fitted_values'], self.res_df['residuals']
            
            smoothed = lowess(self.res_df['residuals'],self.res_df['fitted_values'])
            
            ax.scatter(X, Y, color='r', linewidth = 3, alpha = 0.85, s = 6)
            
            ax.plot(smoothed[:,0], smoothed[:,1], color = 'k')
            
            ax.set_ylabel('Residuals')
            
            ax.set_xlabel('Fitted Values')
            
            ax.set_title('Residuals vs. Fitted')
            
            ax.plot([min(X),max(X)], [0,0], color = 'k', linestyle = ':', alpha = .8)
            
            for i in top3.index:
                ax.annotate(i, xy=(X[i], Y[i]), fontsize = 10, color = 'k', fontweight = 'bold')
        
        if n == 2:

            Y = self.res_df['residuals']
           
            # qq plot from statsmodels.api = sm
            QQplot = sm.qqplot(Y, fit = True, line = "q",ax = ax)

            QQplot.axes[0].set_title('Normal Q-Q')

            QQplot.axes[0].set_xlabel('Theoretical Quantiles')

            QQplot.axes[0].set_ylabel('Standardized Residuals')

            plt.show()

            # axes = plt.gca()

            # x_values, y_values = axes.lines[0].get_xdata(), axes.lines[0].get_ydata()

            # for i, j in enumerate(np.arange(3)-3):
                
            #     indx = np.argsort(np.abs(Y)).values[j]
                
            #     QQplot.axes[0].annotate(
            #         indx, xy = (x_values[j], y_values[j]), fontsize = 10, color = 'k', fontweight = 'bold')
                                    
        if n == 3:
            
            X, Y = self.res_df['fitted_values'], self.res_df['sqrt_student_residuals']

            smoothed = lowess(Y,X)

            # rows with top residuals 
            top3 = abs(self.res_df['residuals']).sort_values(ascending = False)[:3]
            
            ax.scatter(X, Y, alpha = 0.85, linewidths = 3, color = 'r', s = 6)
            
            ax.plot(smoothed[:,0],smoothed[:,1],color = 'k');
            
            ax.set_ylabel('$\sqrt{Studentized \ Residuals}$', fontsize = 12, fontweight = 'bold')
            
            ax.set_xlabel('Fitted Values', fontsize = 12, fontweight = 'bold')
            
            ax.set_title('Scale-Location', fontsize = 14, fontweight = 'bold')
            
            ax.set_ylim(0, max(Y)+0.1)
            
            for i in top3.index:
                ax.annotate(i, xy = (X[i],Y[i]), fontsize = 10, color = 'k', fontweight = 'bold')
        
        if n == 4:  
            
            df_res = self.res_df.reset_index().drop('index', axis=1)
            
            X, Y = df_res.index, df_res['cooks_distance']
            
            # rows with the highest cook distance
            top3 = abs(df_res['cooks_distance']).sort_values(ascending = False)[:3]

            # creates cordinates for vertical lines
            lines = []
            
            for i in range(len(X)):
                
                pair = [(X[i],0), (X[i], Y[i])]
                
                lines.append(pair)
            
            linecoll = matcoll.LineCollection(lines)
            
            ax.add_collection(linecoll)
            
            ax.scatter(X,Y, alpha = 0.85, linewidths = 3, color = 'r', s = 6)
            
            ax.set_ylabel("Cook's distance", fontsize = 12, fontweight = 'bold')
            
            ax.set_xlabel('Observation number', fontsize = 12, fontweight = 'bold')

            for i in top3.index:
                ax.annotate(i, xy = (X[i],Y[i]), fontsize = 10, color = 'k', fontweight = 'bold')

            plt.show()

        if n == 5: 

            student_residuals = pd.Series(self.mdl.get_influence().resid_studentized_internal)
            
            student_residuals.index = self.mdl.resid.index
            
            df = pd.DataFrame(student_residuals)
            
            df.columns = ['student_residuals']

            df['leverage'] = self.mdl.get_influence().hat_matrix_diag
            
            # top 3
            sorted_student_residuals = abs(df['student_residuals']).sort_values(ascending = False)
            
            top3 = sorted_student_residuals[:3]

            # cordinates
            x, y = df['leverage'], df['student_residuals']

            smoothed = lowess(y, x)
            
            ax.scatter(x, y, linewidths = 3, color = 'r', s = 6)
            
            ax.plot(smoothed[:,0], smoothed[:,1], color = 'k')
            
            ax.set_ylabel('Studentized Residuals')
            
            ax.set_xlabel('Leverage')
            
            ax.set_title('Residuals vs. Leverage')
            
            ax.set_ylim(min(y)-abs(min(y))*0.25, max(y)+max(y)*0.15)
            
            ax.set_xlim(-0.01,max(x)+max(x)*0.05)
            
            plt.tight_layout()
            
            for val in top3.index:
                ax.annotate(val, xy = (x.loc[val],y.loc[val]), fontsize = 10, color = 'k', fontweight = 'bold')

            xpos = max(x) + max(x)*0.01
            
            cooksx, p  =  np.linspace(min(x), xpos, 50), len(self.mdl.params)

            poscooks1y, negcooks1y = np.sqrt((p*(1-cooksx))/cooksx), -np.sqrt((p*(1-cooksx))/cooksx)
            
            poscooks05y, negcooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx), -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

            ax.plot(cooksx, poscooks1y, label = "Cook's Distance", ls = ':', color = 'r')
            
            ax.plot(cooksx, poscooks05y, ls = ':', color = 'r')
            
            ax.plot(cooksx, negcooks1y, ls = ':', color = 'r')
            
            ax.plot(cooksx, negcooks05y, ls = ':', color = 'r')
            
            ax.plot([0,0], ax.get_ylim(), ls=":", alpha = .3, color = 'k')
            
            ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
            
            ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
            
            ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
            
            ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
            
            ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
            
            ax.legend()


        if n == 6: 

            X, Y = self.res_df['leverage'], self.res_df['cooks_distance']

            smoothed = lowess(Y,X)
            
            # rows with highest cooks distance
            top3 = abs(Y).sort_values(ascending = False)[:3]

            ax.scatter(X, Y, alpha = 0.85, linewidths = 3,color = 'r', s = 6)
            
            ax.plot(smoothed[:,0],smoothed[:,1],color = 'k')
            
            ax.set_ylabel("Cook's distance")
            
            ax.set_xlabel('Leverage')
            
            ax.set_title("Cook's dist vs. Leverage")
           
            for i in top3.index:
                ax.annotate(i, xy = (X.loc[i],Y.loc[i]), fontsize = 10, color = 'k', fontweight = 'bold')

        
        if n == 7: 
            
            self.df_params.plot(x = 'var', y = 'coef', kind = 'barh', ax = ax, color = 'none', 
                           ecolor = self.df_params.Significant.map({True: '#20B2AA', False: '#F08080'}),
                           capsize = 0, xerr = 'errors', legend = False)

            ax.scatter(y = np.arange(self.df_params.shape[0]), marker = 'o', s = 160, x = self.df_params['coef'], 
                       color = self.df_params.Significant.map({True: '#20B2AA', False: '#F08080'}))

            ax.axvline(x = 0, linestyle = '--', color = '#F08080', linewidth = 1)

            ax.set_title('95% confidence intervals')

            ax.set_ylabel('Coefficients')

            ax.set_xlabel('estimates')
            
            ax.xaxis.grid(False)

        return fig


def repr_table(models, model_number=False):
    """
    Function for displaying models from statsmodels in one table
    
    Args:
        models (list) list of models to display 
        
    Returns:
        object: table displaying all models
    """
    results = Stargazer(models)
    
    model_names = [f"model_{str(i).zfill(2)}" for i in range(1, len(models)+1)]
    
    model_index = [1 for i in range(1, len(models)+1)]

    results.custom_columns(model_names, model_index)

    # remove model numbers
    if not model_number:
        results.show_model_numbers(False)
        
    return results
