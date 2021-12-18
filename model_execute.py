from datetime import datetime
from pmdarima.arima import auto_arima
from datetime import datetime
from pandas import DataFrame
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from matplotlib.pyplot import fill_between
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from numpy import arange


class Model_execute:
    """
    Class responsible for estimating the model.

    Required settings:
    - daframe (input data)
    - variable (formatted dependent variable - "NAME VARIABLE")
     
    Syntax: descriptive_statistics.(data frame, variable name)
    """

    def __init__(self, data, variable):
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
            self.color3 = "darkorange"
            self.color4 = "black"


    def auto_arima(self, s):
        """
        Function that uses auto arima to find the best parameters for the model.
        """
        
        # Best model with auto arima
        model_select = auto_arima(self.data_select,
                                  seasonal=True,
                                  error_action="ignore",
                                  supress_warnings=True,
                                  trace=False)
        
        # Data frame
        model_select = str(model_select)
        paramet = [datetime.now(),
                   int(model_select[7]),
                   int(model_select[9]),
                   int(model_select[11]),
                   int(model_select[14]),
                   int(model_select[16]),
                   int(model_select[18])]
        
        df_paramet = DataFrame(paramet).T
        
        df_paramet.columns=["time", "p", "d", "q", "P", "D", "Q"]
        
        df_paramet["s"] = s
        
        df_paramet.to_csv("3_working/auto_arima_parameters.csv",
                          decimal=".",
                          sep=",",
                          index=False)
        
        return


    def model_execute(self, p, d, q, P, D, Q, s):
        """
        Model estimation.
        
        p = Order of the AR term
        q = Order of the MA term
        d = Number of differencing required to make the time series stationary
        P = Seasonal order of the AR term
        D = Seasonal order of the MA term
        Q = Seasonal difference number
        """
        
        model = ARIMA(self.data_select, order=(p, d, q), 
                            seasonal_order=(P, D, Q, s))
        
        self.model_fit = model.fit()
        model_result = self.model_fit.summary()
        
        with open('4_results/9_model_summary.txt', 'w') as desc_stat:
            desc_stat.write(str(model_result))
        
        return

    def acf_pacf_residuals(self):
        """
        Residuals ACF and PACF.
        """
        
        fig, ax = plt.subplots(2, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        resid = DataFrame(self.model_fit.resid, columns=[f"{self.variable}"])
        
        acf = plot_acf(resid.values.squeeze(),
                lags = len(resid) / 3,
                use_vlines = True,
                title = f"{self.variable} - ACF (RESIDUALS)",
                color = self.color1,
                vlines_kwargs = {"colors": self.color1},
                alpha=0.05,
                zero=False,
                ax=ax[0])
        
        pacf = plot_pacf(resid.values.squeeze(),
                lags = len(resid) / 3,
                use_vlines = True,
                title = f"{self.variable} - PACF (RESIDUALS)",
                color = self.color2,
                vlines_kwargs = {"colors": self.color2},
                alpha=0.05,
                zero=False, 
                ax=ax[1])
        
        ax[0].set_ylim(-0.3, 0.3)
        ax[1].set_ylim(-0.4, 0.3)
        plt.tight_layout()
        
        plt.savefig(f"4_results/10_residuals (acf and pacf) - "
                                        f"{self.variable_}.jpg")
        
        return


    def residuals_analysis(self):
        """
        Analysis of model residuals.
        """
        
        fig, ax = plt.subplots(1, 1, sharex=True, dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        resid = DataFrame(self.model_fit.resid, columns=[f"{self.variable}"])
        
        resid_plot_fd = resid.hist(color=self.color2,
                                   legend=False,
                                   density=True,
                                   figsize=(12, 6))

        plt.title(f"{self.variable.upper()} - RESIDUALS")
        
        plt.tight_layout()
        
        plt.savefig(f"4_results/11_residuals (frequency distribution) - " 
                    f"{self.variable_}.jpg")
        
        return


    def ts_residuals_plot(self):
        """
        Residuals time serie plot.
        """
        
        resid = DataFrame(self.model_fit.resid, columns=[f"{self.variable}"])

        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        resid_plot = resid.plot(title=f"RESIDUALS - {self.variable}",
                                color=self.color2,			
                                legend=False,
                                xlabel="",
                                ylabel="")

        plt.tight_layout()
        resid_plot.figure.savefig(f"4_results/12_residuals (time serie) - "
                          f"{self.variable_}.jpg")
        
        return


    def adjust_predict(self):
        """
        Effective x fitted + predict plot.
        """
        
        # plot config
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        # fit model
        self.data_select[f"{self.variable_}_fitted"] = self.model_fit.predict(start=30,
                                                               end=len(self.data_select),
                                                               dynamic=False)
        
        # plot observed
        effective = self.data_select.iloc[ :, 0 ].plot(title=f"{self.variable}",
                                                            xlabel="",
                                                            ylabel="",
                                                            color=self.color1,
                                                            figsize=(12, 6))
        
        # plot fitted
        adjustment = self.data_select.iloc[ : , 1 ].plot(xlabel="",
                                                    ylabel="",
                                                    color=self.color2)
        
        # forecast
        forecast = self.model_fit.get_forecast(10)
        
        # predict
        predict = forecast.predicted_mean
        
        # confidence interval
        predict_conf_int_95 = forecast.conf_int(alpha=0.05)
        predict_conf_int_50 = forecast.conf_int(alpha=0.50)
        
        #plot predict
        predict = predict.plot( xlabel="",
                                ylabel="",
                                color=self.color3)
        
        #plot confidence interval
        predict_conf_95 = fill_between(predict_conf_int_95.index,
                                       predict_conf_int_95.iloc[ : , 0 ],
                                       predict_conf_int_95.iloc[ : , 1 ],
                                       color=self.color4,
                                       alpha=0.05)

        predict_conf_50 = fill_between(predict_conf_int_50.index,
                                       predict_conf_int_50.iloc[ : , 0 ],
                                       predict_conf_int_50.iloc[ : , 1 ],
                                       color=self.color4, 
                                       alpha=0.1)

        plt.legend(["original",
                    "fitted_model",
                    "forecast",
                    "int. 95%",
                    "int. 50%"])
        
        # save
        plt.tight_layout()
        plt.savefig(f"4_results/13_effective_fitted - "
                                 f"{self.variable_}.jpg")
        
        df_predict = DataFrame(forecast.predicted_mean)
        
        df_predict = df_predict.rename(columns={'predicted_mean': f"{self.variable_}_forecast"})
        
        df_predict.to_csv('4_results/14_predict_mean.txt',
                          sep=",",
                          decimal=".",
                          index_label="index_date")
        
        return

