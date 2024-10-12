import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tools
lat, lon = 61.211093, 23.762815


temperatures = tools.get_current_temperature(lat, lon)
hourly_prices = tools.get_raw_electricity_prices(start='2024-10-12')#, end='2024-02-01')
hourly_prices = tools.prepare_electricity_data(hourly_prices)

tools.compare_price_with_off_hours(hourly_prices, hourly_consumption=0.25, off_hours=8)

temp_power, poly_func = tools.get_pump_power(plot=True)

heatpump_data = tools.prepare_heatpump_data(hourly_prices, temperatures, poly_func)

tools.clock_plot(heatpump_data)
