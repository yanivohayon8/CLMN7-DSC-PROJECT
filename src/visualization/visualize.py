import seaborn as sns
import matplotlib.pyplot as plt
import os

import pandas as pd

class MyPlotting():
    
    @staticmethod
    def print_stats(mydata,df_stats_similarity,title="default title",figure_path=None):
        
        desc = df_stats_similarity.describe() 
        print(desc)
        
        _min = df_stats_similarity.min().values[0]
        _max = df_stats_similarity.max().values[0]
        _median = df_stats_similarity.median().values[0]
        
        fig,axs = plt.subplots(1,2)
        
        fig.suptitle(title)
        
        axs[0].hist(mydata)
        
        axs[1] = sns.heatmap(
        mydata, 
        vmin=_min, vmax=_max, center=_median,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
        )
        #axs[1].set_title(title)
        axs[1].set_xticklabels(
            axs[1].get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        
        if figure_path is not None:
            fig_path = os.path.join(figure_path,title + '.png')
            plt.savefig(fig_path)
        
        plt.show()
    
    @staticmethod
    def similarity_matrix(mydata,min_=0,max_=1,center=0.5,title="default title",figure_path=None):
        
        fig,axs = plt.subplots(1,2)
        
        axs[0].hist(mydata)
        
        axs[1] = sns.heatmap(
        mydata, 
        vmin=min_, vmax=max_, center=center,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
        )
        axs[1].set_title(title)
        axs[1].set_xticklabels(
            axs[1].get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        
        if figure_path is not None:
            fig_path = os.path.join(figure_path,title + '.png')
            plt.savefig(fig_path)
        
        plt.show()
        