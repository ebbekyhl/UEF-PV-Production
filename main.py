import download
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = True
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)

download_dir = os.path.relpath("data")  # choose where to save data

month_mapping = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
                    6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
                    11: "Nov", 12: "Dec"}


# Get the date of today
today = pd.Timestamp.now()

print("Today is the " + str(today.day) + " of " + month_mapping[today.month])
print("Last month was " + month_mapping[today.month - 1])
months = np.arange(1, today.month)
year_months = [f"2025-{month:02d}" for month in months]

for year_month in year_months:
    download.download_pv_data(year_month,download_dir)

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

plt.rcParams.update({
    "text.usetex": True,   # Use LaTeX for text rendering
    "font.family": "serif",  # Use a serif font (you can choose other options)
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional, for math symbols
})

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
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")

# set "date" as index
df.set_index("date", inplace=True, drop=True)

# df comes in daily values, here we convert it to monthly values
production_monthly_sum = df.groupby(df.index.month).sum() # sum of production per month
production_monthly_mean = df.groupby(df.index.month).mean() # mean of production per month

production_monthly_sum.index = production_monthly_sum.index.map(month_mapping)

########################################################################################
########################### Daily production plot ######################################
########################################################################################

fig, ax = plt.subplots(figsize=(8, 5))
df.plot(ax=ax, legend=False, color="darkorange")

ax.set_title("Daily production values")
ax.set_ylabel("MWh")
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
plt.tight_layout()

# savefig
fig.savefig(download_dir + "/daily_production_" + year_month + ".png", bbox_inches='tight')

########################################################################################
########################### Monthly production plot ####################################
########################################################################################

# initialize figure
fig_m, ax_m = plt.subplots(figsize=(8, 5))

# bar plot of monthly production
production_monthly_sum.plot(kind="bar", ax=ax_m, color="darkorange", alpha=0.7, edgecolor="k", width=0.8)

# add expected production from simulation
pd.Series(pvgis).loc[production_monthly_sum.index].plot(marker="X", ls="--", color="k", alpha=0.6, label="Simulation", ax=ax_m)

# figure formatting
ax_m.grid()
ax_m.set_title("Monthly production values (" + year_months[-1][0:4] + ")")
ax_m.set_ylabel("MWh")
ax_m.set_xlabel("")
ax_m.set_xticklabels(ax_m.get_xticklabels(), rotation=0, ha='center')
ax_m.legend()

# savefig
plt.tight_layout()
fig_m.savefig(download_dir + "/monthly_production_" + year_month + ".png", bbox_inches='tight')