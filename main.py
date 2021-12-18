import data_input_config
from descriptive_statistics import Time_serie_level
from x13arima_seas_adjust import X13_arima_desaz
from stationarity import Stationarity_diff
from model_execute import Model_execute
from pandas import read_csv


# data input and configs
# ==========================================================================
data = data_input_config.data_select
variable = data_input_config.variable
p_value_accepted = data_input_config.p_value_accepted
path_x13_arima = data_input_config.path_x13_arima
data_non_seasonal = data_input_config.data_non_seasonal
# ==========================================================================

# level descriptive statistics
# ==========================================================================
descriptive_statistics = Time_serie_level(data, variable)

descriptive_statistics.time_serie_plot()
descriptive_statistics.acf_pacf_plot()
descriptive_statistics.periodogram_plot()
descriptive_statistics.descriptive_stat()
# ==========================================================================

# X13-ARIMA-SEATS
# ==========================================================================
x13_desaz = X13_arima_desaz(data, variable, path_x13_arima)

x13_desaz.x13_results()
x13_desaz.x13_seasonal_adjustment()
# ==========================================================================

# stationarity
# ==========================================================================
stationarity = Stationarity_diff(data_non_seasonal, variable, p_value_accepted)

stationarity.adf_teste()
stationarity.diff_data()
# ==========================================================================

# auto arima parameters
# ==========================================================================
model = Model_execute(data_non_seasonal, variable)

model.auto_arima(data_input_config.s)

# parameters
try:
    auto_arima_paramet = read_csv("3_working/auto_arima_parameters.csv",
                                 index_col="time")

    p = auto_arima_paramet["p"][-1]
    d = auto_arima_paramet["d"][-1]
    q = auto_arima_paramet["q"][-1]
    P = auto_arima_paramet["P"][-1]
    D = auto_arima_paramet["D"][-1]
    Q = auto_arima_paramet["Q"][-1]
    s = auto_arima_paramet["s"][-1]

except:
    p = data_input_config.p
    d = data_input_config.d
    q = data_input_config.q
    P = data_input_config.P
    D = data_input_config.D
    Q = data_input_config.Q
    s = data_input_config.s

model.model_execute(p, d, q, P, D, Q, s)
model.residuals_analysis()
model.acf_pacf_residuals()
model.ts_residuals_plot()
model.adjust_predict()
# ==========================================================================

