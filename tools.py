import io
from pytz import timezone
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from globals import API_KEY

def get_current_temperature(lat, lon):
    complete_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'

    # Sending request to the URL
    response = requests.get(complete_url)

    # Checking if request was successful
    if response.status_code == 200:
        data = response.json()
        print(f'{data["city"]["name"]}, {data["city"]["country"]}')
        times, temps, humidities = [], [], []
        for item in data['list']:
            times.append(item['dt_txt'])
            temps.append(item['main']['temp'])
            humidities.append(item['main']['humidity'])
        # Extracting the temperature
        dat = pd.DataFrame({'time': times, 'temp': temps, 'humidity': humidities})
        dat['datetime_helsinki'] = pd.to_datetime(dat['time'], utc=True).dt.tz_convert(timezone('Europe/Helsinki'))
        return dat
    else:
        return None


def get_raw_electricity_prices(start='2023-09-01', end=None):
    url = f'https://sahkotin.fi/prices.csv?fix&vat&start={start}T00:00:00.000Z'
    if end is not None:
        url = f'{url}&end={end}T00:00:00.000Z'
    x = requests.get(url)
    df = pd.read_csv(io.StringIO(x.text))
    return df


def prepare_electricity_data(df, additional_cost=0.6):
    df['datetime_utc'] = pd.to_datetime(df['hour'], utc=True)
    df['datetime_helsinki'] = df['datetime_utc'].dt.tz_convert(timezone('Europe/Helsinki'))
    df = df.set_index('datetime_helsinki')

    df['year'] = df.index.strftime('%Y')
    df['month'] = df.index.strftime('%m')
    df['date'] = df.index.strftime('%d')
    df['hour'] = df.index.strftime('%H')
    df['day_of_week'] = df.index.strftime('%w').astype(int)
    df['day_of_week'] = df['day_of_week'].replace(0, 7)
    df['price'] = df['price'] + additional_cost
    return df


def daily_sum_excluding_top_n(df, n):
    """
    Takes a DataFrame with hourly datetime index and prices, returns the daily sum
    excluding the three most expensive hours per day.

    Parameters:
    df (pd.DataFrame): DataFrame with hourly datetime index and prices.

    Returns:
    pd.Series: Daily sum excluding the top 3 most expensive hours.
    """

    # Function to calculate the sum excluding top 3 prices
    def sum_excluding_top3(prices):
        # Sort the prices and exclude the 3 highest
        return prices.nsmallest(len(prices) - n).sum()

    # Resample the DataFrame to daily frequency and apply the custom sum function
    daily_sum = df.resample('D').apply(sum_excluding_top3)

    return daily_sum


def compare_price_with_off_hours(hourly_prices, off_hours, hourly_consumption):
    increased_power = round(hourly_consumption * 24 / (24 - off_hours), 2)
    all_day = ((daily_sum_excluding_top_n(hourly_prices['price'], n=0) * 0.25).sum()/100).round(2)
    excluded = ((daily_sum_excluding_top_n(hourly_prices['price'], n=off_hours) * increased_power).sum()/100).round(2)

    diff = round(all_day - excluded, 2)

    print(f'Total price with {hourly_consumption} kW all day: {all_day} €')
    print(f'With {off_hours} hours off, power should be {increased_power} kW, and the total sum is {excluded} €')
    print(f'Difference is {diff} €, {round(diff/all_day*100, 1)} %')


def get_pump_power(plot=True):
    temperature = [-32.5, -30, -25, -20, -15, -10, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12]
    power = [1.6, 2, 2.2, 2.35, 2.65, 2.95, 3.3, 3.6, 4.1, 4.8, 5.45, 6.1, 6.1, 5]

    # Perform 6th-degree polynomial fitting
    coefficients = np.polyfit(temperature, power, 6)

    # Create a polynomial function from the coefficients
    poly_func = np.poly1d(coefficients)

    # Generate new temperature points at every 0.1 degrees
    temperature_new = np.arange(min(temperature), max(temperature), 0.1)

    # Use the polynomial function to compute new power values
    power_new = poly_func(temperature_new)

    if plot:
        # Optional: Plot the original and interpolated data to visualize
        plt.figure(figsize=(8, 4))
        plt.plot(temperature, power, 'o', label='Luettu speksistä')  # Original data
        plt.plot(temperature_new, power_new, '-', label='Sovite')  # Interpolated data
        plt.title('Lämpöpumpun lämmöntuoton tehokerroin ulkolämpötilan mukaan')
        plt.xlabel('Ulkolämpötila (°C)')
        plt.ylabel('Lämpöpumpun tehokerroin')
        plt.legend()
        plt.grid()
        plt.show()

    temp_power = dict(zip(temperature_new, power_new))
    return temp_power, poly_func


def prepare_heatpump_data(hourly_prices, temperatures, poly_func):
    hourly_prices = hourly_prices.merge(temperatures, on='datetime_helsinki')
    hourly_prices['power'] = [poly_func(temp) for temp in hourly_prices['temp']]
    hourly_prices['kilowatteja eurolla'] = hourly_prices['power'] / hourly_prices['price'] * 100
    hourly_prices['skaalattu'] = hourly_prices['price'] / hourly_prices['power']

    plt.figure(figsize=(10, 5))
    lwd = 2
    plt.plot(hourly_prices['datetime_helsinki'], hourly_prices['temp'], linewidth=lwd, label='Ulkolämpötila (°C)')
    plt.plot(hourly_prices['datetime_helsinki'], hourly_prices['power'], linewidth=lwd, label='Lämpöpumpun tehokerroin')
    plt.plot(hourly_prices['datetime_helsinki'], hourly_prices['price'], linewidth=lwd, label='Syöttösähkön hinta')
    plt.plot(hourly_prices['datetime_helsinki'], hourly_prices['skaalattu'], linewidth=lwd, label='Tuotetun lämmön hinta (snt/kWh)')
    plt.title('Tuotetun lämmön hinta sekä siihen vaikuttavat tekijät')
    plt.grid()
    plt.legend()
    plt.show()
    return hourly_prices


def clock_plot(hourly_prices, plot_raw_prices=False):
    # Convert time to angle in radians (0 to 2*pi for 24 hours)
    time_in_seconds = (hourly_prices['datetime_helsinki'].dt.hour * 3600)
    angles = 2 * np.pi * time_in_seconds / (24 * 3600)  # Normalize to [0, 2*pi]

    # Get price as radial coordinate
    radii = hourly_prices['skaalattu'].values
    radii2 = hourly_prices['price'].values

    # Plot in polar coordinates
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, radii, label='')
    if plot_raw_prices:
        ax.plot(angles, radii2, label='Syöttösähkö')

    min_price_indices = np.where(radii == radii.min())[0]
    for i in min_price_indices:
        ax.plot(
            [angles[i - 1] if i > 0 else 0, angles[i], angles[i + 1]],
            [radii[i - 1] if i > 0 else 0, radii[i], radii[i + 1]],
            color='lightgreen', linewidth=3, label='Halvimmat\ntunnit')

    ax.annotate('Nyky-\nhetki', (angles[0], radii[0]), textcoords='offset points', xytext=(5, 5), ha='right')

    # Optional: Set labels and grid
    ax.set_title('Tuotetun lämmön hinta kellonajan mukaan, snt/kWh', va='bottom')
    ax.set_theta_zero_location("N")  # Set 12 o'clock as the top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([f'Klo {str(i * 3)}' for i in range(8)])
    plt.legend()
    plt.tight_layout()
    plt.show()




