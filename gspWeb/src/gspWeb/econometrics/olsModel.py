import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as dg
from scipy.stats import shapiro


def get_kidgrp(x,y):
    if x+y == 0:
        return "no kids"
    elif x ==0 and y>0:
        return "kids aged 6+ years"
    elif x >0 and y == 0:
        return "kids below 6 years"
    else:
        return "kids below & above 6 years"
    
def get_data():
    
    df = pd.read_stata("http://fmwww.bc.edu/ec-p/data/wooldridge/mroz.dta")
    dropCols = ["inlf", "hours", "repwage", "hushrs", "huswage", "faminc", "mtr", "nwifeinc"]
    features = [col for col in df.columns if col != "lwage" and col not in dropCols]

    df = (df
        .query("inlf==1")
        .drop(dropCols, axis=1)
        .assign(
            educgr = lambda X: pd.cut(X["educ"], bins = [4,11,13, 18], labels=('Diploma','Degree','Masters') ,ordered=True),
            agegr = lambda X: pd.cut(X["age"], bins=[0, 35, 45, X["age"].max() + 1e-15 ], labels=["less than 35", "less than 45", "45+"]),
            expergr = lambda X: pd.cut(X["exper"], bins=[0, 5, 10, 15, X["exper"].max() + 1e-15 ], labels=["less than 5", "less than 10", "less than 15", "15+"]),
            kidstl = lambda X: X["kidslt6"] + X["kidsge6"],
            kids = lambda X: np.where(X["kidstl"]>0, 1, 0),
            kidsgr = lambda X: [get_kidgrp(x,y) for x, y in zip(X["kidslt6"], X["kidsge6"])],
        )
        .assign(
            kids = lambda X: X["kids"].astype("category"),
            city = lambda X: X["city"].astype("category")
        )

    )
    df[features] = df[features].astype(int)
    
    return df

def plot_model_summary(mdl):
    fig = plt.figure(figsize=(1,0.1))
    plt.text(x=0.0, y=1, 
        s=str(mdl.summary()), 
        fontdict = {'fontsize': 10}, 
        fontproperties = 'monospace'
    )
    plt.axis('off')
    plt.tight_layout()
    return fig

def ols_diagnostic_test(mdl):
    
    reset_pvalue = dg.linear_reset(mdl, power = [2,3], test_type = 'fitted', use_f = True).pvalue
    
    statistic, shapiro_pvalue = shapiro(mdl.resid)
    
    bp_stat, breushPagan_pvalue, _, _ = dg.het_breuschpagan(mdl.resid, mdl.model.exog)
    
    
    return reset_pvalue, shapiro_pvalue, breushPagan_pvalue




class ols_model:
    def __init__(self):
        
        self.df = get_data()

        self.colDetails = {
            "number of kids below 6 years":{
                "colName":"kidslt6",
                "uniqVal":[0, 1, 2],
                "labels":[0, 1, 2],
                "ylabel":"kids",
                "xlabel":"wage"
            },

            "number of kids 6+ years":{
                "colName":"kidsge6",
                "uniqVal":[0, 1, 2, 3, 4, 5, 8],
                "labels":[0, 1, 2, 3, 4, 5, 8],
                "ylabel":"kids",
                "xlabel":"wage"
            },

            "education":{
                "colName":"educgr",
                "uniqVal":["Diploma", "Degree", "Masters"],
                "labels":["Diploma", "Degree", "Masters"],
                "ylabel":"education",
                "xlabel":"wage"
            },

            "age":{
                "colName":"agegr",
                "uniqVal":["less than 35", "less than 45", "45+"],
                "labels":["less than 35", "less than 45", "45+"],
                "ylabel":"age",
                "xlabel":"wage"
            },

            "experience":{
                "colName":"expergr",
                "uniqVal":["less than 5", "less than 10", "less than 15", "15+"],
                "labels":["less than 5", "less than 10", "less than 15", "15+"],
                "ylabel":"experience",
                "xlabel":"wage(log)"
            },

            "lives in city":{
                "colName":"city",
                "uniqVal":[0, 1],
                "labels":["no", "yes"],
                "ylabel":"city status",
                "xlabel":"wage"
            },

            "parent":{
                "colName":"kids",
                "uniqVal":[0, 1],
                "labels":["no", "yes"],
                "ylabel":"parent",
                "xlabel":"wage"
            },

            "parent category":{
                "colName":"kidsgr",
                "uniqVal":["no kids", "kids below 6 years", "kids aged 6+ years", "kids below & above 6 years"],
                "labels":["no kids", "kids below 6 years", "kids aged 6+ years", "kids below & above 6 years"],
                "ylabel":"parent category",
                "xlabel":"wage"
            },
        }
        
    def plot_details(self, description):
        self.description = description
        self.colDetail = self.colDetails.get(description)
        self.col = self.colDetail.get("colName")
        self.uniqVals = self.colDetail.get("uniqVal")
        self.labels = self.colDetail.get("labels")
        self.ylabel = self.colDetail.get("ylabel")
        self.xlabel = self.colDetail.get("xlabel")
        self.target = "wage"
        self.histData = []
        for uniqVal in self.uniqVals:
            data = self.df.query(f"{self.col}==@uniqVal")[self.target].values
            self.histData.append(data)
        self.colors = ["#ff2e63", "deepskyblue", "#01949A"]
        
    def customize_fig(self, fig):
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor="#979797",
                linewidth=1,
                ticks="outside",
                title=f"{self.xlabel}"
            ),
            yaxis=dict(
                showline=False,
                showgrid=True,
                showticklabels=False,
                gridcolor="lightgray",
                title=f"{self.ylabel}"
            ),

            legend=dict(
                orientation="h",
                x=0.5,
                y=1.1
            ),

            font=dict(
                size=12,
                color="navy"
            ),

            title=dict(
                text=f"Distribution of wage by {self.description}",
                font=dict(
                    size=20,
                    color="navy"
                )
                ,x=0.5,
                xanchor="center"
            ),
        )
        
        return fig
    
    def ols_distplot(self, description="education"):
        self.plot_details(description)
        
        fig = ff.create_distplot(
            hist_data = self.histData,
            group_labels = [str(x) for x in self.labels],
            bin_size=0.5,
            curve_type="kde",
            colors=self.colors,
            show_hist=False,
            show_curve=True,
            show_rug=False,
        )
        
        fig = self.customize_fig(fig)
        return fig
        
    def ols_violinplot(self, description="education"):
        
        self.plot_details(description)
        
        fig = go.Figure()

        if len(self.labels)>3:
            for data_line, label in zip(self.histData, self.labels):
                fig.add_trace(go.Violin(x=data_line, name=str(label)))
        else:
            for data_line, color, label in zip(self.histData, self.colors, self.labels):
                fig.add_trace(go.Violin(x=data_line, line_color=color, name=str(label)))

        fig.update_traces(orientation='h', side='positive', width=3, points=False, meanline_visible=True)
        
        fig = self.customize_fig(fig)
        return fig
    
    def get_ols(self):
        formula = "lwage ~ exper + educ + age + kidslt6 + kidsge6 + unem + city"
        self.olsModel = smf.ols(formula, data=self.df).fit()
        self.olsModel_reset_pvalue, self.olsModel_shapiro_pvalue, self.olsModel_breushPagan_pvalue = ols_diagnostic_test(self.olsModel)

        return self.olsModel

    def get_olsExt(self):
        formula = "lwage ~ exper + I(exper**2) + educ + age + kidslt6 + kidsge6 + unem + city "
        self.olsModelExt = smf.ols(formula, data=self.df).fit() 
        self.olsModelExt_reset_pvalue, self.olsModelExt_shapiro_pvalue, self.olsModelExt_breushPagan_pvalue = ols_diagnostic_test(self.olsModelExt)
        return self.olsModelExt
    
    def plot_exp(self):
        df_exp = pd.DataFrame( {'experience':np.arange(50)})
        
        if not hasattr(self, "olsModelExt"):
            self.get_olsExt()

        # predict wage using experience as the only variable
        df_exp['log(wage)'] = (
            df_exp['experience']
            .apply(
                lambda x: self.olsModelExt.params['Intercept'] + self.olsModelExt.params['exper']*x 
                + self.olsModelExt.params['I(exper ** 2)']*(x**2) + self.olsModelExt.params['educ']*12.6 
                + self.olsModelExt.params['age']*42 + self.olsModelExt.params['kidslt6']*0.1402 
                + self.olsModelExt.params['kidsge6']*42 + self.olsModelExt.params['unem']*8.3 
                + self.olsModelExt.params['city']*1
            )
        )

        fig = px.line(df_exp, x="experience", y="log(wage)", markers=True, color_discrete_sequence=['deepskyblue'])
        fig.update_layout(
            width=1000, 
            height=600,
            plot_bgcolor="white",
            xaxis=dict(
                showline=True,
                zeroline=False,
                showgrid=False,
                showticklabels=True,
                linecolor="#979797",
                linewidth=1,
                ticks="outside",
                title=f"experience"
            ),
            yaxis=dict(
                showline=False,
                showgrid=True,
                showticklabels=True,
                gridcolor="lightgray",
                title=f"log(wage)"
            ),

            legend=dict(
                orientation="h",
                x=0.5,
                y=1.1
            ),

            font=dict(
                size=12,
                color="navy"
            ),

            title=dict(
                text=f"Effect of wage on experience",
                font=dict(
                    size=20,
                    color="navy"
                )
                ,x=0.5,
                xanchor="center"
            ),
        )
        
        return fig
    
    def plot_exp_animated(self):

        df_exp = pd.DataFrame( {'experience':np.arange(50)})
        
        if not hasattr(self, "olsModelExt"):
            self.get_olsExt()

        # predict wage using experience as the only variable
        df_exp['log(wage)'] = (
            df_exp['experience']
            .apply(
                lambda x: self.olsModelExt.params['Intercept'] + self.olsModelExt.params['exper']*x 
                + self.olsModelExt.params['I(exper ** 2)']*(x**2) + self.olsModelExt.params['educ']*12.6 
                + self.olsModelExt.params['age']*42 + self.olsModelExt.params['kidslt6']*0.1402 
                + self.olsModelExt.params['kidsge6']*42 + self.olsModelExt.params['unem']*8.3 
                + self.olsModelExt.params['city']*1
            )
        )
        x = df_exp.experience
        y = df_exp["log(wage)"]

        # Create the initial trace
        trace = go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(width=2, color='deepskyblue')
        )

        # Create the animation frames
        frames = [go.Frame(data=[go.Scatter(x=x[:i], y=y[:i], mode='lines+markers')]) for i in range(len(x))]

        # Add the frames to the layout
        layout = go.Layout(
            width=1000, 
            height=600,
            plot_bgcolor="white",
            xaxis=dict(
                range=[-1, 50],
                showline=True,
                zeroline=False,
                showgrid=False,
                showticklabels=True,
                linecolor="#979797",
                linewidth=1,
                ticks="outside",
                title=f"experience"
            ),
            yaxis=dict(
                range=[0.2, 1],
                showline=False,
                showgrid=True,
                showticklabels=True,
                gridcolor="lightgray",
                title=f"log(wage)"
            ),
            
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=120), fromcurrent=True, transition=dict(duration=0))]
                )]
            )],
            
            legend=dict(
                orientation="h",
                x=0.5,
                y=1.1
            ),
            
            font=dict(
                size=12,
                color="navy"
            ),
            
            title=dict(
                text=f"Effect of experience on wage",
                font=dict(
                    size=20,
                    color="navy"
                )
                ,x=0.5,
                xanchor="center"
            ),
        )

        fig = go.Figure(data=[trace], layout=layout, frames=frames)

        return fig
    
