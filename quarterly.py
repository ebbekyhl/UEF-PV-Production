import os
import pandas as pd
import numpy as np

download_dir = os.path.abspath("data")  # choose where to save data

month_mapping = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Maj",
                    6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt",
                    11: "Nov", 12: "Dec"}

month_mapping_long = {1: "Januar", 2: "Februar", 3: "Marts", 4: "April", 5: "Maj",
                    6: "Juni", 7: "Juli", 8: "August", 9: "September", 10: "Oktober",
                    11: "November", 12: "December"}

# Get the date of today
today = pd.Timestamp.now()

print("Today is the " + str(today.day) + " of " + month_mapping[today.month])
print("Last month was " + month_mapping[today.month - 1])
print("The month before that was " + month_mapping[today.month - 2])
print("And the month before that was " + month_mapping[today.month - 3])

months = np.arange(1, today.month)
year_earliest = 2025
year_today = today.year
year_months = [f"{year_today}-{month:02d}" for month in months]
if year_earliest < year_today:
    years = np.arange(year_earliest, year_today)
    for year in years:
        year_months += [f"{year}-{month:02d}" for month in np.arange(1, 13)]

# read data for every months contained in "year_months"
df = pd.concat([pd.read_csv(download_dir + f"/PV_production_Aarhus_{month}.csv", sep=";") for month in year_months])
df.rename(columns={"Ep":"Produktion"}, inplace=True)

# convert "date" into pd.datetime
df["date"] = pd.to_datetime(df["date"])

# set "date" as index
df.set_index("date", inplace=True, drop=True)

# df comes in daily values, here we convert it to monthly values
production_monthly_sum = df.groupby(df.index.month).sum() # sum of production per month
production_monthly_mean = df.groupby(df.index.month).mean() # mean of production per month

production_monthly_sum.index = production_monthly_sum.index.map(month_mapping)

production_last_month = production_monthly_sum.iloc[-1].item()  # get the last month production values
production_two_months_ago = production_monthly_sum.iloc[-2].item()  # get the production values for two months ago
production_three_months_ago = production_monthly_sum.iloc[-3].item()  # get the production values for three months ago

def format_number(number):
    formatted_number = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted_number

def extract_month_name(year_month):
    return month_mapping_long[int(year_month.split("-")[1])]

def extract_year(year_month):
    return year_month.split("-")[0]

with open("email_summary.txt", "w", newline="") as f:
    f.write(f"Samlet produktion for {extract_month_name(year_months[-3])} {extract_year(year_months[-3])}: {format_number(production_three_months_ago)} kWh\r\n")
    f.write(f"Samlet produktion for {extract_month_name(year_months[-2])} {extract_year(year_months[-2])}: {format_number(production_two_months_ago)} kWh\r\n")
    f.write(f"Samlet produktion for {extract_month_name(year_months[-1])} {extract_year(year_months[-1])}: {format_number(production_last_month)} kWh\r\n")