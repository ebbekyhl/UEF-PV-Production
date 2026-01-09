import os
import pandas as pd
import numpy as np
from daily import get_year_months

download_dir = os.path.abspath("data")  # choose where to save data

month_mapping = {-2: "Oct", -1: "Nov", 0: "Dec",
                 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Maj",
                 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt",
                 11: "Nov", 12: "Dec"}

month_mapping_long = {-2: "Oktober", -1: "November", 0: "December",
                      1: "Januar", 2: "Februar", 3: "Marts", 4: "April", 5: "Maj",
                      6: "Juni", 7: "Juli", 8: "August", 9: "September", 10: "Oktober",
                      11: "November", 12: "December"}

# Get the date of today
today = pd.Timestamp.now()
print("Today is the " + str(today.day) + " of " + month_mapping[today.month])
print("Last month was " + month_mapping[today.month - 1])
print("The month before that was " + month_mapping[today.month - 2])
print("And the month before that was " + month_mapping[today.month - 3])

year_months = get_year_months(today)

########################################################################################
############################# Data for the last quarter ################################
########################################################################################
df = pd.read_csv(download_dir + "/PV_monthly_production_full_period.csv")

production_last_month = df.iloc[-2]["Produktion"]  # get last month's production values
production_two_months_ago = df.iloc[-3]["Produktion"]  # get production values from two months ago
production_three_months_ago = df.iloc[-4]["Produktion"]  # get production values from three months ago

def format_number(number):
    formatted_number = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted_number

def extract_month_name(year_month):
    return month_mapping_long[int(year_month.split("-")[1])]

def extract_year(year_month):
    return year_month.split("-")[0]

with open("email_summary.txt", "w", newline="") as f:
    f.write(f"Samlet produktion for {extract_month_name(year_months[-4])} {extract_year(year_months[-4])}: {format_number(production_three_months_ago)} kWh\r\n")
    f.write(f"Samlet produktion for {extract_month_name(year_months[-3])} {extract_year(year_months[-3])}: {format_number(production_two_months_ago)} kWh\r\n")
    f.write(f"Samlet produktion for {extract_month_name(year_months[-2])} {extract_year(year_months[-2])}: {format_number(production_last_month)} kWh\r\n")