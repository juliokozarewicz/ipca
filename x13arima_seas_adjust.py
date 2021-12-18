from statsmodels.tsa.x13 import x13_arima_analysis as x13a
from pandas import DataFrame
from pandas import read_csv
from matplotlib import pyplot as plt
from sys import platform


class X13_arima_desaz:
    """
    X-13 ARIMA-SEATS, successor to X-12-ARIMA and X-11, is a set of statistical 
    methods for seasonal adjustment and other descriptive analysis of time 
    series data that are implemented in the U.S. Census Bureau's.

    Required settings:
    - daframe (input data)
    - variable (formatted dependent variable - "NAME VARIABLE")
    - path (Directory of the folder where x13 arima seats are located)
    
    Syntax: descriptive_statistics.(data frame, variable name, path)
    """

    def __init__(self, data, variable, path):
        """
        Settings for the outputs.
        """
        
        # data frame
        self.data_select = data
        
        # configs
        self.variable = variable
        self.variable_ = variable.replace(" ", "_").lower()
        
        # style
        self.style_graph = "seaborn"
        self.color1 = "royalblue"
        self.color2 = "crimson"
        
        # X13-ARIMA-SEATS CONFIG
        self.x13_desaz = x13a(data, x12path=path)


    def x13_results(self):
        """
        Results obtained with X13-ARIMA-SEATS
        """
        
        # style
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        # x13 results
        x13_results_plot = self.x13_desaz.plot()
        plt.tight_layout()
        x13_results_plot.savefig(f'4_results/7_x13_results_{self.variable_}.jpg')
        
        return


    def x13_seasonal_adjustment(self):
        """
        X13 Seasonal adjustment.
        """
        
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        x13_seasonal = DataFrame(self.x13_desaz.seasadj.values,
                                 index=self.data_select.index.values,
                                columns=[self.variable_])
        
        x13_seasonal_plot_raw = plt.plot(self.data_select,
                                         color=self.color1,
                                         label="original")
        
        x13_seasonal_plot = plt.plot(x13_seasonal,
                                     color=self.color2,
                                     label="seasonal adjustment")
        
        plt.legend(loc="upper right")
        plt.title("X13-ARIMA SEASONAL ADJUSTMENT")
        plt.tight_layout()
        plt.savefig(f'4_results/8_x13_seasonal_adjustment_{self.variable_}.jpg')
        
        # new data frame
        x13_seasonal.to_csv("3_working/seasonal_adjustment.csv",
                               index_label="index_date", sep=";", decimal=".")
        
        return

