import seaborn as sns
import matplotlib.pyplot as plt
import os

class MyPlotting():
    @staticmethod
    def similarity_matrix(mydata,title,figure_path):
        ax = sns.heatmap(
        mydata, 
        vmin=0, vmax=1, center=0.5,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
        )
        ax.set_title(title)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        
        if figure_path is not None:
            fig_path = os.path.join(figure_path,title + '.png')
            plt.savefig(fig_path)
        
        plt.show()