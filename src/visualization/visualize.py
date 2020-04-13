import seaborn as sns
class MyPlotting():
    @staticmethod
    def similarity_matrix(mydata,title):
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