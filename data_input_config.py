from pandas import DataFrame
from pandas import to_datetime
from pandas import read_csv


# data input
# ==========================================================
data_entry = read_csv("3_working/data_base.csv", sep=";", decimal=",")
data_entry["index_date"] = to_datetime(data_entry["index_date"])
data_select = data_entry.sort_values("index_date")

# date filter
data_select = data_select[(data_select.iloc[:,0] >= '2015-01') &
                         (data_entry.iloc[:,0] <= '2021-12') ]

# variable for executing the model estimation
data_select = data_select.set_index("index_date").iloc[ : , 0:1 ]
# ==========================================================

# seasonally adjusted data entry
# ==========================================================
try:
    data_non_seasonal = read_csv("3_working/seasonal_adjustment.csv",
                                     sep=";", decimal=".")
    data_non_seasonal["index_date"] = to_datetime(data_non_seasonal["index_date"])
    data_non_seasonal = data_non_seasonal.sort_values("index_date")

    # variable for executing the model estimation
    data_non_seasonal = data_non_seasonal.set_index("index_date")

except:
    data_non_seasonal = data_select
# ==========================================================

# variables
# ==========================================================
variable_ = list(data_select.columns.values.tolist())[0]
variable = variable_.replace("_", ' ').upper()
p_value_accepted = 0.05
path_x13_arima = "/home/edu/x13as/"
# ==========================================================

# Manual parameter setting
# ==========================================================
p = 1
d = 0
q = 1
P = 0
D = 0
Q = 1

# inform the periodicity of the series (D=365, M=12, Y=1)
s = 12
# ==========================================================

