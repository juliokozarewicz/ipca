from statsmodels.tsa.stattools import adfuller as adf
from pandas import read_csv

class Stationarity_diff:
    """
    Study of data stationarity.
    
    Required settings:
        - daframe (input_data)
        - variable (formatted dependent variable - "NAME VARIABLE")
        - p_value_accepted (p-value number accepted)
        
    Syntax: stationarity.(data frame, variable name)
    """

    def __init__(self, data, variable, p_value_accepted):
        """
        Settings for the outputs.
        """
        
        # data frame
        self.data_select = data
        
        # configs
        self.variable = variable
        self.variable_ = variable.replace(" ", "_").lower()
        self.p_value_accepted = p_value_accepted
        
        # style
        self.style_graph = "seaborn"
        self.color1 = "royalblue"
        self.color2 = "crimson"


    def adf_teste(self):
        """
        Adf test.
        """
        
        adf_level = adf(self.data_select, regression='ct')

        adf_level_result = (
        f"{'-' * 50}\n"
        f"ADF Results (level):\n\n"
        f"Variable: {self.variable}\n"
        f"ADF Test: {adf_level[0]:.6f}\n"
        f"P-value: {adf_level[1]:.6f}\n"
        f"Lags: {adf_level[2]}\n"
        f"Observations: {adf_level[3]}\n"
        f"Critical values:\n"
        f"  1%: {adf_level[4]['1%']:.6f}\n"
        f"  5%: {adf_level[4]['5%']:.6f}\n"
        f"  10%: {adf_level[4]['10%']:.6f}\n"
        f"{'-' * 50}")
        
        with open('4_results/5_adf_test_level.txt', 'w') as desc_stat:
            desc_stat.write(adf_level_result)
        
        return


    def diff_data(self):
        """"
        Function that returns the stationary series through the ADF test criterion 
        and differentiation method. The function will also set the value of 
        parameter (d).
        """
        
        count_diff = 0
        
        while True:
            adf_test_diff = adf(self.data_select, regression='ct')
            adf_p_value = adf_test_diff[1]
            
            if adf_p_value > self.p_value_accepted:
                stationary_series = self.data_select.diff().dropna()
                self.data_select = stationary_series
                count_diff += 1
                self.data_select.to_csv("3_working/stationary_db.csv")
            
            else:
                adf_diff = adf(self.data_select, regression='ct')
                
                adf_result = (
                f"{'-' * 50}\n"
                f"ADF Results:\n\n"
                f"Nonseasonal differences needed for stationarity: {count_diff}\n\n"
                f"Variable: {self.variable}\n"
                f"ADF Test: {adf_diff[0]:.6f}\n"
                f"P-value: {adf_diff[1]:.6f}\n"
                f"Lags: {adf_diff[2]}\n"
                f"Observations: {adf_diff[3]}\n"
                f"Critical values:\n"
                f"  1%: {adf_diff[4]['1%']:.6f}\n"
                f"  5%: {adf_diff[4]['5%']:.6f}\n"
                f"  10%: {adf_diff[4]['10%']:.6f}\n"
                f"{'-' * 50}")
                
                with open('4_results/6_adf_diff_result.txt', 'w') as desc_stat:
                    desc_stat.write(adf_result)
                
                break
        
        return
