import download
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

download_dir = download_dir = os.path.abspath("data")  # choose where to save data

month_mapping = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
                    6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
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

# list the files that are already downloaded 
existing_files = os.listdir(download_dir)

# missing files 
year_months_downloads = [ym for ym in year_months if f"PV_production_Aarhus_{ym}.csv" not in existing_files]

print(year_months_downloads)

for year_month_download in year_months_downloads:
    download.download_pv_data(year_month_download,download_dir)

# For comparison, we show the expected production from simulation:
pvgis = {"Jan": 895.7,
        "Feb": 2037.71,
        "Mar": 5641.7,
        "Apr": 9541.1,
        "May": 12132.2,
        "Jun": 12686.7,
        "Jul": 11868.2,
        "Aug": 9501.8,
        "Sep": 6703.4,
        "Oct": 3242.3,
        "Nov": 1108.7,
        "Dec": 562.9}

# plotting configuration
fs = 14
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['axes.titlesize'] = fs + 2
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['legend.title_fontsize'] = fs+3
plt.rcParams['legend.fontsize'] = fs

# read data for every months contained in "year_months"
df = pd.concat([pd.read_csv(download_dir + f"/PV_production_Aarhus_{month}.csv", sep=";") for month in year_months])
df.rename(columns={"Ep":"Production"}, inplace=True)

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

########################################################################################
########################### Daily production plot ######################################
########################################################################################

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(ax=ax, legend=False, color="darkorange")

ax.set_title("Daily production values")
ax.set_ylabel("kWh")
ax.set_xlabel("")
ax.grid()

# add a double-directed arrow just below the graph of the last month 
arrowprops = dict(arrowstyle="<->", lw=2, color="gray")

subtract = 1/len(year_months)  # calculate the width of the arrow based on the number of months
y_mean = production_monthly_mean.iloc[-1].item()
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
y_text = 1 - y_mean/y_range
x_text = (2 - subtract)/2 

ax.annotate("", xy=(1.0, y_text), xytext=(1.0 - subtract, y_text), 
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=arrowprops)

text = production_monthly_sum.index[-1] + " " +  year_months[-1][0:4]

ax.text(x_text, y_text - y_range*0.00001, text, ha='center', va='top', transform=ax.transAxes, fontsize=fs, color="k", alpha=0.75)

# savefig
fig.savefig("figures/production_" + year_months[-1] + "_daily.png")

########################################################################################
########################### Monthly production plot ####################################
########################################################################################

# initialize figure
fig_m, ax_m = plt.subplots(figsize=(8, 5))

# bar plot of monthly production
production_monthly_sum.plot(kind="bar", ax=ax_m, color="darkorange", alpha=0.7, edgecolor="k", width=0.8)

# add expected production from simulation
pd.Series(pvgis).loc[production_monthly_sum.index].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet", ax=ax_m)

# figure formatting
ax_m.grid()
ax_m.set_title("Månedlige produktionsværdier (" + year_months[-1][0:4] + ")")
ax_m.set_ylabel("kWh")
ax_m.set_xlabel("")
ax_m.set_xticklabels(ax_m.get_xticklabels(), rotation=0, ha='center')
ax_m.legend()

# savefig
fig_m.savefig("figures/production_" + year_months[-1] + "_monthly.png")