create_rule.py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.display import display

from optbinning import OptimalBinning
from optbinning import BinningProcess

params = {
    "figure.figsize":(10,6),
    "text.color":"#5b68f6",
    "axes.titlesize":16,
    "axes.labelsize":14,
    "axes.labelcolor": "#5b68f6",
    "axes.edgecolor": "#5b68f6",
    "xtick.color": "#5b68f6",
    "ytick.color": "#5b68f6",
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "legend.fontsize":12, 
    "axes.grid.axis":"y",
    "axes.spines.left":False,
    "axes.spines.top":False,
    "axes.spines.right":False,
    "axes.edgecolor": "#5b68f6",
}

plt.rcParams.update(params)


def plot_woe_bins(binning_result, column_name):
    
    woe_table = binning_result.get_binned_variable(column_name).binning_table.build()
    
    display(woe_table)
    
    woe_table.drop(['Totals'], inplace=True)

    woe_table['WoE'] = pd.to_numeric(woe_table['WoE'])

    woe_table = woe_table.query("abs(WoE)>0")

    woe_table['Bin'] = woe_table['Bin'].astype('string')
    
    fig, ax = plt.subplots(figsize=(9,5))

    sns.barplot(x='Bin', y='WoE', data=woe_table, ax=ax, color='deepskyblue')
    
    ax.grid(axis="y", linestyle="-", color="lightgray", zorder=0)
    ax.set_axisbelow(True)

    # ax.set_title(f'WoE per bins created for {column_name}')

    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.tick_params(axis='x', which='major', rotation=90)

    ax.set_xlabel('Bin', fontsize=10)

    ax.set_ylabel('WoE', fontsize=10)

    ax.set_xlabel('Bin', fontsize=10)
    
    ax.axhline(0, color='blue', linewidth=1)
    
    ax.spines['bottom'].set_edgecolor('lightgrey')
    
    plt.show()
    
    
class DiscretizeFeature:
    
    @staticmethod
    def get_quantile_bins(data, value_column, target='target', q=10, divisor = 1):
        
        results = (
            data
            .assign(score_bucket = lambda X: pd.qcut(X[value_column]/divisor, q=q, duplicates='drop'))
            .groupby(['score_bucket'])
            .agg(
                lower_bound=(value_column, 'min'),
                upper_bound=(value_column, 'max'),
                count=(target, 'count'),
                events=(target, 'sum'),
                event_rate=(target, 'mean'),
            )
            .assign(
                non_events = lambda X: X['count'] - X['events'],
                distr = lambda X: X['count']/X['count'].sum(),
                target_rate = lambda X: X['events']/X['events'].sum(),
                # lower_bound = lambda X: [round(x/10, 0)*10 if x>=10 else round(x, 0) for x in X['lower_bound']],
                # upper_bound = lambda X: [round(x/10, 0)*10 if x>=10 else round(x, 0) for x in X['upper_bound']],
            )
            .filter(items=['lower_bound', 'upper_bound', 'count', 'non_events', 'events', 'distr', 'target_rate',  'event_rate'])
        )
        
        return results
    
    @staticmethod
    def get_quantile_plots(data, ax, title='Quantile distributions', table=True, **kwargs):
        
        results = DiscretizeFeature.get_quantile_bins(
            data, 
            value_column=kwargs.get('value_column'), 
            target=kwargs.get('target', 'default_target'), 
            q=kwargs.get('q', 10), 
            divisor=kwargs.get('divisor', 1)
        )
        
        if table:
            display(results)

        results.index = results.index.astype(str)

        ax.bar(results.index, results['events'], color='#ff2e63')

        ax.bar(results.index, results['non_events'], bottom=results['events'], color='deepskyblue')

        ax.tick_params(axis='x', rotation=90)

        ax.grid(axis="y", linestyle="-", color="lightgray", zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.set_tick_params(width=0, length=0)

        ax2 = ax.twinx()

        ax2.plot(results['event_rate'], marker='o', color='#0000FF', label='event_rate')

        ax2.plot(results['distr'], marker='o', color='black', label='distr')

        ax2.plot(results['target_rate'], marker='o', color='red', label='target_rate')

        ax2.set_yticklabels(['{:.0%}'.format(x) for x in ax2.get_yticks()])

        ax2.tick_params(axis='y', right=False)

        ax2.grid(False)

        ax2.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncols=3, frameon=False)

        ax.set_xlabel('Bins', fontsize=10)

        plt.show()
     
    @staticmethod
    def get_custom_thresholds(data, value_column, thresholds, target='target', logic='less than', divisor=1, as_int = False, **kwargs):
        
        if logic == 'less than':
            logic_sign = '<'
        else:
            logic_sign = '>'
            
        thresholds = sorted(thresholds)

        bin_results = (
            data
            .merge(pd.Series(thresholds, name="bucket"), how="cross")
            .assign(bucket = lambda X: X['bucket'].astype(float))
            .assign(**{value_column: lambda X: X[value_column]/divisor})
            .query(f"{value_column} {logic_sign} bucket")
            .groupby(["bucket"], as_index=False)
            .agg(
                lower_bound=(value_column, 'min'),
                count=(target, 'count'),
                events=(target, 'sum'),
                event_rate=(target, 'mean'),
            )
        )
        
        if not as_int:
            bin_results['bucket'] = bin_results['bucket'].apply(lambda x: f'{x:.2f}')
        else:
            bin_results['bucket'] = bin_results['bucket'].astype(int).astype(str)
        
        return bin_results
    
    @staticmethod
    def get_custom_thresholds_plot(bin_results, threshold=None, logic='', title='Distribution', width=1000,):
        
        df_plot = bin_results.copy()
        
        df_plot['bucket'] = df_plot['bucket'].str.replace('.00', '')
    
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Bar(x=df_plot['bucket'], y=df_plot['count'], marker=dict(color="deepskyblue"), name="Count", showlegend=True), secondary_y=False)

        fig.add_trace(go.Scatter(x=df_plot['bucket'], y=df_plot['event_rate'], marker=dict(color="red", size=8), mode='lines+markers', line_shape='linear', name="Event rate", showlegend=True), secondary_y=True)
        
        if threshold:
            index = bin_results.query(f'bucket==@threshold').index.values[0]
            fig.add_shape(
                type="line",
                x0=index,
                x1=index,
                y0=0,
                y1=max(bin_results["count"]),  
                line=dict(color="blue", width=1, dash="dash") 
            )


        fig.update_layout(
            width=width,
            plot_bgcolor="white",
            font=dict(size=12, color="#5b68f6"),
            title=dict(text=f"{title}", x=0.5, xanchor="center", font=dict(size=16, color="#5b68f6")),
            yaxis=dict(showline=True, showticklabels=True, showgrid=True, gridcolor="lightgray", title="Count"),
            xaxis=dict(showgrid=False, linecolor="#5b68f6", linewidth=1, ticks="outside", title=f"Threshold (logic = {logic})", tickfont=dict(size=10)),
            yaxis2=dict(title="Event rate", showgrid=True, tickformat=".1%"),
            legend=dict(x=0.4, y=1.1, orientation="h", title_text=""),
            legend_tracegroupgap=100,
            bargroupgap=0.1,
            bargap=0.2
        )

        fig.show("png")