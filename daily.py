import download
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import requests

use_inverter_data = True # whether to use inverter data if available

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
months = np.arange(1, today.month+1)
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

# always include the latest month in "year_months_downloads"
if year_months[-1] not in year_months_downloads:
    year_months_downloads.append(year_months[-1])

for year_month_download in year_months_downloads:
    download.download_pv_data(year_month_download,download_dir)

# For comparison, we show the expected production from simulation:
pvgis = {"Jan": 895.7,
        "Feb": 2037.71,
        "Mar": 5641.7,
        "Apr": 9541.1,
        "Maj": 12132.2,
        "Jun": 12686.7,
        "Jul": 11868.2,
        "Aug": 9501.8,
        "Sep": 6703.4,
        "Okt": 3242.3,
        "Nov": 1108.7,
        "Dec": 562.9}

# For now, assume the following self-consumptions:
self_consumption_ratio = pd.Series({"Jan": 1,
                          "Feb": 0.95,
                          "Mar": 0.9,
                          "Apr": 0.88,
                            "Maj": 0.85,
                            "Jun": 0.85,
                            "Jul": 0.85,
                            "Aug": 0.9,
                            "Sep": 0.95,
                            "Okt": 0.97,
                            "Nov": 1,
                            "Dec": 1
                            } # read from plot by Parisa
                            )

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

def format_number(number):
    formatted_number = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted_number

def extract_month_name(year_month):
    return month_mapping_long[int(year_month.split("-")[1])]

def extract_year(year_month):
    return year_month.split("-")[0]

def get_values():

    owner = "martavp"
    repo = "UEF"
    path = "data/inverter_monthly_datafiles"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url)

    files = response.json()

    files_list = []
    for file in files:
        files_list.append(file["name"])

    data_dir = "https://raw.githubusercontent.com/martavp/UEF/main/data/inverter_monthly_datafiles/"
    # https://github.com/martavp/UEF/blob/main/data/inverter_monthly_datafiles/Inverter_2_2024_09.xlsx
    # https://raw.githubusercontent.com/martavp/UEF/main/data/inverter_monthly_datafiles/Inverter_2_2024_09.xlsx

    existing_files = [f"data/inverter_data/{f}" for f in os.listdir("data/inverter_data/") if f.endswith(".csv")]

    df_daily_production_inv = {}
    for file in files_list:

        # check if file exists in data already - if so, read it
        if f"data/inverter_data/{file}.csv" in existing_files:
            df_inv_hourly = pd.read_csv(f"data/inverter_data/{file}.csv", index_col=0, parse_dates=True)
            df_inv = df_inv_hourly.resample("d").sum() 
            inverter = file.split("_")[1]
            month = df_inv.index[0].month
            year = df_inv.index[0].year
            ym = f"{year}-{month:02d}"
            df_daily_production_inv[(inverter, ym)] = df_inv
            continue

        inverter = file.split("_")[1]

        df_inv = pd.read_excel(data_dir + file, skiprows=[0,1,2], header = 0)

        df_inv["Start Time"] = df_inv["Start Time"].str.split(" ", expand=True)[0] + " " + df_inv["Start Time"].str.split(" ", expand=True)[1]
        df_inv["time"] = pd.to_datetime(df_inv["Start Time"])
        df_inv.set_index("time", inplace=True)

        month = df_inv.index[0].month
        year = df_inv.index[0].year

        ym = f"{year}-{month:02d}"

        df_hourly_production_inv = df_inv["Active power(kW)"].resample("h").mean()

        # Saving file 
        print(f"Saving inverter data for inverter {inverter} for {ym}:")
        df_hourly_production_inv.to_csv(f"data/inverter_data/{file}.csv")

        df_daily_production_inv[(inverter, ym)] = df_hourly_production_inv.resample("d").sum() 

    return df_daily_production_inv

# read data for every months contained in "year_months"
df = pd.concat([pd.read_csv(download_dir + f"/PV_production_Aarhus_{month}.csv", sep=";") for month in year_months])
df.rename(columns={"Ep":"Produktion"}, inplace=True)
# convert "date" into pd.datetime
df["date"] = pd.to_datetime(df["date"])

# set "date" as index
df.set_index("date", inplace=True, drop=True)

# make a copy of df
df_copy = df.copy()

###########################################################################################
############################ Inverter data replacement ####################################
###########################################################################################
# some reported values at the AURORA dashboard might be incorrect, so we replace these data 
# points with more accurate inverter data from the GitHub repo: https://github.com/martavp/UEF

if use_inverter_data:
    # download data from GitHub repository
    df_inv = get_values()

    # set index and sum the two inverters
    df_inv_1 = pd.concat(df_inv).loc["1"].reset_index()
    df_inv_2 = pd.concat(df_inv).loc["2"].reset_index()
    df_inv_1.set_index("time", inplace=True)
    df_inv_2.set_index("time", inplace=True)
    common_index = df_inv_1.index.intersection(df_inv_2.index)
    df_inv_sum = df_inv_1.loc[common_index, "Active power(kW)"] + df_inv_2.loc[common_index, "Active power(kW)"]

    # create dataframe with inverter data where available, NaN elsewhere
    df_copy["Replaced"] = False
    common_index2 = df_copy.index.intersection(df_inv_sum.index)
    df_copy.loc[common_index2, "Produktion"] = df_inv_sum
    df_copy.loc[common_index2, "Replaced"] = True
    df_inverter_data_index = df_copy.query("Replaced == True")["Produktion"].index
    df_non_inverter_data_index = df_copy.drop(df_inverter_data_index).index
    df_inverter_data = df_copy["Produktion"].copy()
    df_inverter_data.loc[df_non_inverter_data_index] = np.nan

    # swap values where inverter data is larger than the original data
    df_final = df["Produktion"].copy()
    swap_index = df_inverter_data.index[df_inverter_data > df.loc[df_inverter_data.index, "Produktion"]]
    df_final.loc[swap_index] = df_inverter_data.loc[swap_index]
    df["Produktion"] = df_final

########################################################################################
############################# Monthly values ###########################################
########################################################################################
# df comes in daily values, here we convert it to monthly values
production_monthly_sum = df.groupby(df.index.month).sum() # sum of production per month
production_monthly_mean = df.groupby(df.index.month).mean() # mean of production per month

production_monthly_sum.index = production_monthly_sum.index.map(month_mapping)
# production_monthly_sum.columns = ["Produktion " + year_months[-1][0:4]]

production_monthly_sum_cum = production_monthly_sum.cumsum()

production_last_month = production_monthly_sum.iloc[-1].item()  # get the last month production values

with open("email_summary.txt", "w", newline="") as f:
    f.write(f"Samlet produktion for {extract_month_name(year_months[-1])} {extract_year(year_months[-1])}: {format_number(production_last_month)} kWh\r\n")

########################################################################################
########################### Read grid data #############################################
########################################################################################
start = '2025-01-01'
today = pd.Timestamp.now()
end = str(today.year) + "-" + str(today.month).zfill(2) + "-" + (str(today.day).zfill(2))
url_emissions = f'https://api.energidataservice.dk/dataset/DeclarationProduction?start={start}&end={end}&filter=' + '{"PriceArea":["DK1"]}'
url_prices = f'https://api.energidataservice.dk/dataset/Elspotprices?start={start}&end={end}&filter=' + '{"PriceArea":["DK1"]}'

# from url_emissions, we can calculate CO2 emissions offset
g_emissions = pd.read_json(url_emissions)
g_emissions = pd.json_normalize(g_emissions["records"])
g_emissions.index = pd.to_datetime(g_emissions["HourDK"])
df_emissions = g_emissions[['ProductionType', 'CO2PerkWh', 'Production_MWh']]
df_emissions.sort_index(inplace=True)
df_emissions_tCO2 = df_emissions["CO2PerkWh"]*df_emissions["Production_MWh"]*1000 # gCO2
df_emissions_production = df_emissions["Production_MWh"]*1000 # kWh
df_emissions_tCO2_d = df_emissions_tCO2.resample("d").sum()
df_emissions_production_d = df_emissions_production.resample("d").sum()
df_emissions_intensity = df_emissions_tCO2_d / df_emissions_production_d # gCO2/kWh 

# from url_prices, we can calculate savings for AU
g_prices = pd.read_json(url_prices)
g_prices = pd.json_normalize(g_prices["records"])
g_prices.index = pd.to_datetime(g_prices["HourDK"])
g_prices["SpotPriceDKK"] /= 1000  # convert from DKK/MWh to DKK/kWh

energinet_tariffs = (7.2 + 4.3)/100 # DKK/kWh, Energinets systemtarif + Nettarif # https://energinet.dk/el/elmarkedet/tariffer/aktuelle-tariffer/
transport_tariffs_s_low = 12.2/100 # DKK/kWh https://elberegner.dk/guides/hvad-koster-transport-af-el/
transport_tariffs_s_mid = 18.31/100 # DKK/kWh, https://elberegner.dk/guides/hvad-koster-transport-af-el/
transport_tariffs_s_high = 47.60/100 # DKK/kWh, https://elberegner.dk/guides/hvad-koster-transport-af-el/
transport_tariffs_w_low = 12.2/100 # DKK/kWh, https://elberegner.dk/guides/hvad-koster-transport-af-el/
transport_tariffs_w_mid = 36.61/100 # DKK/kWh, https://elberegner.dk/guides/hvad-koster-transport-af-el/
transport_tariffs_w_high = 109.85/100 # DKK/kWh, https://elberegner.dk/guides/hvad-koster-transport-af-el/

spot_prices = pd.DataFrame(g_prices["SpotPriceDKK"]).sort_index()
spot_prices["month"] = spot_prices.index.month
spot_prices["season"] = np.where(spot_prices["month"].isin([10, 11, 12, 1, 2, 3]), "winter", "summer")
# add low level between 00 and 06, mid level between 06 and 17, high level between 17 and 21, mid level between 21 and 24
spot_prices["hour"] = spot_prices.index.hour
spot_prices["level"] = np.where(spot_prices["hour"].between(0, 6), "low", 
                                np.where(spot_prices["hour"].between(6, 17), "mid", 
                                         np.where(spot_prices["hour"].between(17, 21), "high", "mid")))

w_low_index = spot_prices.query("season == 'winter' & level == 'low'").index
w_mid_index = spot_prices.query("season == 'winter' & level == 'mid'").index
w_high_index = spot_prices.query("season == 'winter' & level == 'high'").index
s_low_index = spot_prices.query("season == 'summer' & level == 'low'").index
s_mid_index = spot_prices.query("season == 'summer' & level == 'mid'").index
s_high_index = spot_prices.query("season == 'summer' & level == 'high'").index

spot_prices.loc[w_low_index, "transport"] = transport_tariffs_w_low
spot_prices.loc[w_mid_index, "transport"] = transport_tariffs_w_mid 
spot_prices.loc[w_high_index, "transport"] = transport_tariffs_w_high
spot_prices.loc[s_low_index, "transport"] = transport_tariffs_s_low
spot_prices.loc[s_mid_index, "transport"] = transport_tariffs_s_mid
spot_prices.loc[s_high_index, "transport"] = transport_tariffs_s_high

# spot_prices["level"]
g_prices["ElPriceDKK"] = spot_prices["SpotPriceDKK"] + spot_prices["transport"] + energinet_tariffs

carriers = ['BioGas', 'Straw', 'Wood', 'FossilGas', 'Coal', 'Fossil Oil',
              'Waste', 'Hydro', 'Solar', 'WindOffshore', 'WindOnshore']
date_time_series = pd.date_range(start=df_emissions.index.min(), end=df_emissions.index.max(), freq='H')
df_emissions_carrier = pd.DataFrame(index = date_time_series)
for carrier in carriers:
    index = df_emissions.query(f"ProductionType == '{carrier}'")["Production_MWh"].drop_duplicates().index
    # get common index
    common_index = date_time_series.intersection(index)
    df_i = df_emissions.query(f"ProductionType == '{carrier}'")["Production_MWh"].drop_duplicates().loc[common_index]
    df_i = df_i.groupby(df_i.index).sum()
    df_emissions_carrier.loc[common_index, carrier] = df_i.values

colors = {'BioGas': "#fbeb90ff",
          'Straw': "#e2e27c", 
          'Wood': "#88a253",
          'FossilGas': "#564640",
          'Coal': "#000000",
          'Fossil Oil': "#4d4d4d",
          'Waste': "#9F8D71",
          'Hydro': "#37a297",
          'Solar': "#ffce3b",
          'WindOffshore': "#6389f3",
          'WindOnshore': "#1f6cc3"}

names = {'BioGas': "Biogas", 
          'Straw': "Strå", 
          'Wood': "Træ",
          'FossilGas': "Naturgas",
          'Coal': "Kul",
          'Fossil Oil': "Olie",
          'Waste': "Affald",
          'Hydro': "Vandkraft",
          'Solar': "Sol",
          'WindOffshore': "Havvind",
          'WindOnshore': "Landvind"}

########################################################################################
################################## Panel 3 #############################################
########################################################################################

solar_color = "#f9c302" 
blue_color = "#1f77b4"
red_color = "#b93020"

aspect_ratio = 11.69 / 8.27
fig0, ax = plt.subplots(figsize=(10*aspect_ratio,10), nrows = 3, sharex=False)

shares = df_emissions_carrier.sum()/(df_emissions_carrier.sum().sum())*100

df_emissions_carrier = df_emissions_carrier[shares[shares > 0.1].index]

# plot area chart of daily production by carrier in GWh
(df_emissions_carrier.resample("d").sum()/1e3).rename(columns=names).plot.area(stacked=True, 
                                                                               linewidth = 0, 
                                                                               color = [colors[col] for col in df_emissions_carrier.columns], ax=ax[0])

ax[1].fill_between(df_emissions_intensity.index, 0, df_emissions_intensity, color='lightgray', alpha=1, zorder = 0)
ax[1].set_ylim([0, df_emissions_intensity.max()*1.1])
ax[1].set_xticks(ax[0].get_xticks(minor=True), minor=True)
ax[1].set_xticklabels(ax[0].get_xticklabels())

fixed_price = 1.7
price_above = g_prices["ElPriceDKK"] > fixed_price
price_below = g_prices["ElPriceDKK"] <= fixed_price

prices_high = g_prices["ElPriceDKK"].copy()
prices_high.loc[price_below] = np.nan

prices_low = g_prices["ElPriceDKK"].copy()
prices_low.loc[price_above] = fixed_price

ax[2].fill_between(prices_high.index, 
                   fixed_price, 
                   prices_high, 
                   color=blue_color, 
                   lw = 1,
                   alpha=1, 
                   zorder = 0)

ax[2].fill_between(prices_low.index, 
                   0, 
                   prices_low, 
                   color=blue_color, 
                   lw = 0,
                   alpha=0.5, 
                   zorder = 0)

neg_prices = g_prices["SpotPriceDKK"].copy()
neg_prices[neg_prices >= 0] = np.nan 
ax[2].fill_between(neg_prices.index, 
                   0, 
                   neg_prices, 
                   color=red_color, 
                   lw = 1,
                   alpha=1, 
                   zorder = 0)

no_hours_above = prices_high.loc[price_above].shape[0]
ax[2].annotate("Fast pris", 
               xy=(pd.to_datetime("7/7/2025"), 1.9),
               ha='left',
               fontsize=fs,
               color= "k")
ax[2].annotate(f"{no_hours_above} timer over fast pris", 
               xy=(pd.to_datetime("7/7/2025"), 3.5),
               ha='left',
               fontsize=fs,
               color= blue_color)

ax[2].annotate("Negativ rå elpris", 
               xy=(pd.to_datetime("7/1/2025"), -0.55),
               ha='left',
               fontsize=fs,
               color= red_color)

ax[2].set_ylim([-0.7, g_prices["ElPriceDKK"].max()*1.1])
ax[2].set_xticks(ax[0].get_xticks(minor=True), minor=True)
ax[2].set_xticklabels(ax[0].get_xticklabels())
ax[2].axhline(y=fixed_price, color="k", linestyle='--', lw = 1)

# add space between subplots
fig0.subplots_adjust(hspace=0.4)

# remove legend 
ax[0].legend().set_visible(False)
ax[0].set_title(r"$\mathbf{Daglig}$" + " " + r"$\mathbf{elproduktion}$" + " " + r"$\mathbf{i}$" + " " + r"$\mathbf{DK1}$" + " (GWh)", color = "gray") # CO2 udledninger undgået ved egenproduktion
ax[1].set_title(r"$\mathbf{Daglig}$" + " " + r"$\mathbf{CO}_2$" + " " + r"$\mathbf{intensitet}$" + " (gCO$_2$/kWh)", color = "gray")
ax[2].set_title(r"$\mathbf{Spotpris}$" + " + " + r"$\mathbf{nettarif}$" + " (ekskl. elafgift) (DKK/kWh)", color = "gray")

# Layout 
for ax_i in ax:
    # hide upper and right spines
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.grid(lw = 0.5, ls='--', color='gray', alpha=0.7)
    # ax_i.legend(loc = "best")
    ax_i.set_xticklabels(ax_i.get_xticklabels(), rotation=0, ha='center')
    ax_i.set_ylabel("")
    ax_i.set_xlabel("")
    ax_i.set_xlim([df_emissions_carrier.index.min(), df_emissions_carrier.index.max()])

# reverse legend order
handles, labels = ax[0].get_legend_handles_labels()
fig0.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.91, 0.88), loc='upper left', borderaxespad=0.)

# add copyright on bottom right of the figure
fig0.text(0.42, 0.02, '© 2025 Universitetets Energifællesskab (UEF)', ha='right', va='bottom', fontsize=14, color='gray', alpha=0.7)
# savefig
fig0.savefig("figures/production_panel_3.png", bbox_inches='tight')

########################################################################################
################################## Panel 2 #############################################
########################################################################################
fig = plt.figure(figsize=(10*aspect_ratio,10))
gs = fig.add_gridspec(3, 2, height_ratios=[1.4,
                                           0.7,
                                           0.7
                                           ]) 

# add space between subfigures
gs.update(hspace=0.4, wspace=0.15)

# fig, axs = plt.subplots(1, 2, figsize=(14, 5))
ax_m = fig.add_subplot(gs[0,0])
ax_m.set_title(r"$\mathbf{Månedlige}$" + " " + r"$\mathbf{produktionsværdier}$" + " (MWh)", color = "gray")

# bar plot of monthly production
(production_monthly_sum/1e3).plot(kind="bar", ax=ax_m, color=solar_color, alpha=0.7, edgecolor="k", width=0.7, label = "Produktion")
self_consumption = production_monthly_sum["Produktion"]*self_consumption_ratio.loc[production_monthly_sum.index]
(self_consumption/1e3).plot(kind="bar", ax=ax_m, color=solar_color, alpha=0.4, 
                            edgecolor="k", 
                            hatch="/",
                            width=0.7)

# add expected production from simulation
(pd.Series(pvgis)/1e3).loc[production_monthly_sum.index[0:-1]].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet", ax=ax_m)

ax_n = fig.add_subplot(gs[0,1])
ax_n.set_title(r"$\mathbf{Kumuleret}$" + " " + r"$\mathbf{produktion}$" + " (MWh)", color = "gray")

# add expected production from simulation
(pd.Series(pvgis).cumsum()/1e3).loc[production_monthly_sum.index[0:-1]].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet", ax=ax_n)

# cumulative sum of monthly production
(production_monthly_sum_cum/1e3).plot(marker="o", ax=ax_n, color=solar_color, alpha=0.7, lw = 2, zorder = 10, label = "Produktion 2025")
(self_consumption.cumsum()/1e3).plot(ls="-", marker="o", ax=ax_n, color="gray", alpha=0.7, lw = 2, zorder = 5, label = "Egetforbrug 2025")

# Daily production
ax_i = fig.add_subplot(gs[1,:])
ax_i.set_title(r"$\mathbf{Daglig}$" + " " + r"$\mathbf{produktion}$" + " (kWh)", color = "gray")
df.plot(ax= ax_i, lw = 0, alpha=1, color = solar_color)
ax_i.fill_between(df.index, 0, df["Produktion"], color=solar_color, alpha=0.3, zorder = 0)
ax_i.set_ylim(0, ax_i.get_ylim()[1])

# Avoided CO2 emissions
capacity = 100 # kWp
footprint = 385 # kg CO2 per kWp, from https://www.recgroup.com/sites/default/files/documents/wp_-_recs_class-leading_carbon_footprint.pdf?utm_source=chatgpt.com
panel_footprint = capacity * footprint / 1000 # tCO2

ax_k = fig.add_subplot(gs[2,:])
ax_k.set_title(r"$\mathbf{CO}_2$" + " " + r"$\mathbf{regnskab}$" + " (ton)", color = "gray") # CO2 udledninger undgået ved egenproduktion
CO2_avoided = df_emissions_intensity * df["Produktion"]  
net_CO2 = panel_footprint - (CO2_avoided.cumsum()/1e6)

remaining_emissions = net_CO2.dropna().iloc[-1]

co2_avoided_sofar = (CO2_avoided.cumsum()/1e6).dropna().iloc[-1]
no_days_produced = df.shape[0]

co2_avoiding_rate = co2_avoided_sofar / no_days_produced
no_days_remaining = remaining_emissions / co2_avoiding_rate

end_date = df.index.max() + pd.Timedelta(days=no_days_remaining)

net_CO2.plot(ax=ax_k, color='green', lw=1, alpha=0.7, label="Undgået CO2")
ax_k.fill_between(net_CO2.index, 0, net_CO2, color='lightgreen', alpha=0.3, zorder = 0)

ax_k.plot([df.index.max(), end_date], [net_CO2.dropna().iloc[-1], 0], color='green', lw=1, alpha=0.7, ls="--", label="Forventet udvikling")

ax_k.set_ylim(0, net_CO2.max()*1.2)

ax_k.text(pd.to_datetime("2025-01-10"), panel_footprint*1.05, "Fabrik", fontsize=fs, color="gray")
ax_k.text(pd.to_datetime("2029-11-01"), 0, "Break-even", fontsize=fs, color="gray")
ax_k.text(pd.to_datetime("2027-01-01"), 12, "Hvis samme udvikling fortsætter", fontsize=fs, color="green", rotation=-7, alpha =0.5)


ax_k.set_xlim([df.index.min(), pd.to_datetime("2030-06-01")])

# Layout 
for ax in [ax_m, ax_n, ax_i, ax_k]:
    # hide upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(lw = 0.5, ls='--', color='gray', alpha=0.7)
    ax.legend(loc = "best", prop = {'size': fs-2})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_ylabel("")
    ax.set_xlabel("")
    if ax != ax_m and ax != ax_n and ax != ax_k:
        ax.set_xlim([df.index.min(), df.index.max()])

ax_m.legend().set_visible(False)
ax_i.legend().set_visible(False)
ax_k.legend().set_visible(False)

# add hatch to legend in ax_m
handles, labels = ax_m.get_legend_handles_labels()
hatch_handle = plt.Rectangle((0,0),1,1, facecolor="white", edgecolor="k", hatch="//", alpha=0.4)
handles.append(hatch_handle)
labels.append("Egetforbrug")
ax_m.legend(handles[-3:], labels[-3:], 
            bbox_to_anchor=(0.02, 0.98), 
            loc='upper left', 
            borderaxespad=0., 
            prop = {'size': fs-2})

# add copyright on bottom right of the figure
fig.text(0.42, 0.02, '© 2025 Universitetets Energifællesskab (UEF)', ha='right', va='bottom', fontsize=14, color='gray', alpha=0.7)
fig.savefig("figures/production_panel_2.png", bbox_inches='tight')

########################################################################################
############################# Panel 1 ##################################################
########################################################################################
files = os.listdir("data/inverter_data/")
files_list = [f for f in files if f.endswith(".csv")]

df_hourly_production_inv = {}
for file in files_list:
    df_inv_hourly = pd.read_csv(f"data/inverter_data/{file}", index_col=0, parse_dates=True)
    inverter = file.split("_")[1]
    month = df_inv_hourly.index[0].month
    year = df_inv_hourly.index[0].year
    ym = f"{year}-{month:02d}"
    df_hourly_production_inv[(inverter, ym)] = df_inv_hourly

# set index and sum the two inverters
df_inv_1 = pd.concat(df_hourly_production_inv).loc["1"].reset_index()
df_inv_2 = pd.concat(df_hourly_production_inv).loc["2"].reset_index()
df_inv_1.set_index("time", inplace=True)
df_inv_2.set_index("time", inplace=True)
common_index = df_inv_1.index.intersection(df_inv_2.index)
df_inv_sum = df_inv_1.loc[common_index, "Active power(kW)"] + df_inv_2.loc[common_index, "Active power(kW)"]

# only include data from 1st January 2025
df_inv_sum = df_inv_sum.loc["2025-01-01":]

def calculate_pivot(df_inv_sum):
    df_hourly = df_inv_sum.copy()
    df_hourly = df_hourly.to_frame()  # convert Series to DataFrame
    df_hourly['date'] = df_hourly.index.date
    df_hourly['hour'] = df_hourly.index.hour - 1
    df_pivot = df_hourly.pivot_table(values='Active power(kW)', index='hour', columns='date')
    df_pivot = df_pivot.reindex(range(24)).fillna(0)

    df_pivot_min = df_pivot.min(axis=1)
    df_pivot_max = df_pivot.max(axis=1)
    df_pivot_q10 = df_pivot.quantile(0.10, axis=1)
    df_pivot_q25 = df_pivot.quantile(0.25, axis=1)
    df_pivot_q40 = df_pivot.quantile(0.40, axis=1)
    df_pivot_q50 = df_pivot.quantile(0.5, axis=1)
    df_pivot_q60 = df_pivot.quantile(0.6, axis=1)
    df_pivot_q75 = df_pivot.quantile(0.75, axis=1)
    df_pivot_q90 = df_pivot.quantile(0.90, axis=1)

    df_pivot_winter = df_pivot.loc[:, pd.to_datetime(df_pivot.columns).month.isin([12,1,2])]
    df_pivot_summer = df_pivot.loc[:, pd.to_datetime(df_pivot.columns).month.isin([6,7,8])]
    df_pivot_spring = df_pivot.loc[:, pd.to_datetime(df_pivot.columns).month.isin([3,4,5])]
    df_pivot_autumn = df_pivot.loc[:, pd.to_datetime(df_pivot.columns).month.isin([9,10,11])]

    df_pivot_winter_mean = df_pivot_winter.mean(axis=1)
    df_pivot_summer_mean = df_pivot_summer.mean(axis=1)
    df_pivot_spring_mean = df_pivot_spring.mean(axis=1)
    df_pivot_autumn_mean = df_pivot_autumn.mean(axis=1)

    return (df_pivot, df_pivot_min, df_pivot_max, df_pivot_q10, df_pivot_q25, df_pivot_q40,
            df_pivot_q50, df_pivot_q60, df_pivot_q75, df_pivot_q90,
            df_pivot_winter, df_pivot_summer, df_pivot_spring, df_pivot_autumn,
            df_pivot_winter_mean, df_pivot_summer_mean,
            df_pivot_spring_mean, df_pivot_autumn_mean)

# for inverters aggregated
(df_pivot, df_pivot_min, df_pivot_max, df_pivot_q10, df_pivot_q25, df_pivot_q40,
            df_pivot_q50, df_pivot_q60, df_pivot_q75, df_pivot_q90,
            df_pivot_winter, df_pivot_summer, df_pivot_spring, df_pivot_autumn,
            df_pivot_winter_mean, df_pivot_summer_mean,
            df_pivot_spring_mean, df_pivot_autumn_mean) = calculate_pivot(df_inv_sum)

# for inverter 1
(df_pivot_1, df_pivot_min_1, df_pivot_max_1, df_pivot_q10_1, df_pivot_q25_1, df_pivot_q40_1,
            df_pivot_q50_1, df_pivot_q60_1, df_pivot_q75_1, df_pivot_q90_1,
            df_pivot_winter_1, df_pivot_summer_1, df_pivot_spring_1, df_pivot_autumn_1,
            df_pivot_winter_mean_1, df_pivot_summer_mean_1,
            df_pivot_spring_mean_1, df_pivot_autumn_mean_1) = calculate_pivot(df_inv_1["Active power(kW)"])

# for inverter 2
(df_pivot_2, df_pivot_min_2, df_pivot_max_2, df_pivot_q10_2, df_pivot_q25_2, df_pivot_q40_2,
            df_pivot_q50_2, df_pivot_q60_2, df_pivot_q75_2, df_pivot_q90_2,
            df_pivot_winter_2, df_pivot_summer_2, df_pivot_spring_2, df_pivot_autumn_2,
            df_pivot_winter_mean_2, df_pivot_summer_mean_2,
            df_pivot_spring_mean_2, df_pivot_autumn_mean_2) = calculate_pivot(df_inv_2["Active power(kW)"])

def plot_daily_profile(ax,
                       df_pivot,
                       df_pivot_min,
                       df_pivot_max,
                       df_pivot_q10,
                       df_pivot_q25,
                       df_pivot_q40,
                       df_pivot_q50,
                       df_pivot_q60,
                       df_pivot_q75,
                       df_pivot_q90,
                       df_pivot_winter,
                         df_pivot_summer,
                           df_pivot_spring,
                               df_pivot_autumn,
                         df_pivot_winter_mean,
                           df_pivot_summer_mean,
                             df_pivot_spring_mean,
                               df_pivot_autumn_mean,
                        name = "", 
                       ):

   ax.fill_between(df_pivot.index, df_pivot_q10, df_pivot_q90, color=solar_color, alpha=0.2, label="10-90 percentil", lw = 0)
   ax.fill_between(df_pivot.index, df_pivot_q25, df_pivot_q75, color=solar_color, alpha=0.4, label="25-75 percentil", lw = 0)
   ax.fill_between(df_pivot.index, df_pivot_q40, df_pivot_q60, color=solar_color, alpha=0.6, label="40-60 percentil", lw = 0)
   
   lws = {"": 2,
          "Inverter 1": 1,
          "Inverter 2": 1}
   df_pivot_max.plot(color="gray", legend=False, alpha = 1, ax=ax, lw=lws[name], label="Max", ls = ":")
   df_pivot_q50.plot(color="gray", legend=False, alpha = 1, ax=ax, lw=lws[name], label="Median", ls = "--")

   seasons = {
               # "vinter": df_pivot_winter, 
         #    "forår": df_pivot_spring, 
            "sommer": df_pivot_summer, 
         #    "efterår": df_pivot_autumn
            }

   seasons_mean = {
               # "vinter": df_pivot_winter_mean, 
         #    "forår": df_pivot_spring_mean, 
            "sommer": df_pivot_summer_mean, 
         #    "efterår": df_pivot_autumn_mean
            }

   season_colors = {
      "vinter": "#606a71",
      "forår": "#b7ff0e",
      "sommer": "#ffbf00",
      "efterår": "#7b3c3c"
   }
   # for season_name, season in seasons.items():
   #     # season.plot(ax=ax, lw=0.5, label = f"Gennemsnitlig {season_name}dag", color=season_colors[season_name], alpha = 0.5)
   #     seasons_mean[season_name].plot(ax=ax, lw=3, label = f"Gennemsnitlig {season_name}dag", color=season_colors[season_name])

   # plot last day in dataset
   days_back = 1
   df_pivot.iloc[:, -days_back].plot(ax=ax, lw=1, ls="--", color="orange", label="Seneste dag\n(" + str(df_pivot.columns[-days_back]) + ")")

   if name == "":
      ax.set_title(r"$\mathbf{Intradag}$" + " " + r"$\mathbf{effektkurve}$" + " " + name +  " (kW)", color = "gray")
   else: 
      ax.set_title(name +  " (kW)", color = "gray", fontsize = 14)

   # Layout 
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.grid(lw = 0.5, ls='--', color='gray', alpha=0.7)
   ax.legend(loc = "best", prop = {'size': fs-2})
   ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
   ax.set_ylabel("")
   ax.set_xlabel("")

   ax.set_xticks(np.arange(0, 24, 1))

   # get xticks 
   xticks = ax.get_xticks()
   ax.set_xticklabels([f"{i}:00" for i in range(len(xticks))], rotation=45)

   # if inverter 1 or inverter 2, then only show every third xtick label
   if name in ["Inverter 1", "Inverter 2"]:
      ax.set_xticklabels([f"{i}:00" if i % 3 == 0 else "" for i in range(len(xticks))], rotation=30, fontsize = fs - 2)

   ax.legend().set_visible(False)

   if name != "":
      ax.set_ylim(0, 35)

   if name == "":
      # add hatch to legend in ax_m
      handles, labels = ax.get_legend_handles_labels()

      ax.legend(handles, labels, 
                  bbox_to_anchor=(0.02, 0.98), 
                  loc='upper left', 
                  borderaxespad=0., 
                  prop = {'size': fs-2})
      
      ax.set_xlim(0, 23)

fig_hourly = plt.figure(figsize=(14, 6))
gs = fig_hourly.add_gridspec(2, 2, height_ratios=[1,
                                           1,
                                           ], width_ratios=[1.5,
                                                          0.8,
                                                          ]) 

# add space between subfigures
gs.update(hspace=0.4, wspace=0.15)

# fig, axs = plt.subplots(1, 2, figsize=(14, 5))
ax_1 = fig_hourly.add_subplot(gs[0:2,0:1])
ax_2 = fig_hourly.add_subplot(gs[0,1])
ax_3 = fig_hourly.add_subplot(gs[1,1])

plot_daily_profile(ax_1,
                    df_pivot,
                  df_pivot_min,
                  df_pivot_max,
                  df_pivot_q10,
                  df_pivot_q25,
                  df_pivot_q40,
                  df_pivot_q50,
                  df_pivot_q60,
                  df_pivot_q75,
                  df_pivot_q90,
                  df_pivot_winter,
                    df_pivot_summer,
                      df_pivot_spring,
                          df_pivot_autumn,
                    df_pivot_winter_mean,
                      df_pivot_summer_mean,
                        df_pivot_spring_mean,
                          df_pivot_autumn_mean
                  )

plot_daily_profile(ax_2,
                       df_pivot_1,
                       df_pivot_min_1,
                          df_pivot_max_1,
                          df_pivot_q10_1,
                            df_pivot_q25_1,
                            df_pivot_q40_1,
                              df_pivot_q50_1,
                                df_pivot_q60_1,
                                  df_pivot_q75_1,
                                    df_pivot_q90_1,
                            df_pivot_winter_1,
                              df_pivot_summer_1,
                                df_pivot_spring_1,
                                  df_pivot_autumn_1,
                            df_pivot_winter_mean_1,
                              df_pivot_summer_mean_1,
                                df_pivot_spring_mean_1,
                                  df_pivot_autumn_mean_1,
                                  name = "Inverter 1"
                          )

plot_daily_profile(ax_3,
                       df_pivot_2,
                       df_pivot_min_2,
                            df_pivot_max_2,
                            df_pivot_q10_2,
                                df_pivot_q25_2,
                                df_pivot_q40_2,
                                df_pivot_q50_2,
                                    df_pivot_q60_2,
                                    df_pivot_q75_2,
                                        df_pivot_q90_2,
                                df_pivot_winter_2,
                                df_pivot_summer_2,
                                    df_pivot_spring_2,
                                    df_pivot_autumn_2,
                                df_pivot_winter_mean_2,
                                df_pivot_summer_mean_2,
                                    df_pivot_spring_mean_2,
                                    df_pivot_autumn_mean_2,
                                    name = "Inverter 2"
                            )

fig_hourly.text(0.42, -0.05, '© 2025 Universitetets Energifællesskab (UEF)', ha='right', va='bottom', fontsize=14, color='gray', alpha=0.7)
fig_hourly.savefig("figures/production_panel_1.png", bbox_inches='tight')

########################################################################################
############################# Save panel A and B #######################################
########################################################################################
pngs = ["figures/production_panel_1.png",
        "figures/production_panel_2.png", 
        "figures/production_panel_3.png",
        ]

# --- Make a PDF with those PNGs as pages ---
with PdfPages("figures/UEF_rapport.pdf") as pdf:
    for path in pngs:
        img = mpimg.imread(path)
        w,h = 3508, 2480 # A4 at 300dpi
        dpi = 300

        # create a figure at the right size with the correct dpi
        fig_PDF = plt.figure(figsize = (w/dpi, h/dpi), dpi = dpi)
        ax = plt.axes([0, 0, 1, 1])  # full-bleed
        ax.imshow(img)
        ax.axis("off")

        pdf.savefig(fig_PDF)   # embeds the PNG raster on its own PDF page
        plt.close(fig_PDF)

########################################################################################
################################## Panel simple ########################################
########################################################################################
# make panel of figures (two subfigures in the top row and one wide figure in the bottom row)
fig = plt.figure(figsize=(14,8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2,1])  # 2 rows, 2 cols

# add space between subfigures
gs.update(hspace=0.3, wspace=0.15)

# fig, axs = plt.subplots(1, 2, figsize=(14, 5))
ax_m = fig.add_subplot(gs[0,0])
ax_m.set_title("Månedlige produktionsværdier (MWh)")

# bar plot of monthly production
(production_monthly_sum/1e3).plot(kind="bar", ax=ax_m, color=solar_color, alpha=0.7, edgecolor="k", width=0.7)

# add expected production from simulation
(pd.Series(pvgis)/1e3).loc[production_monthly_sum.index].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet", ax=ax_m)

ax_n = fig.add_subplot(gs[0,1])
ax_n.set_title("Kumuleret produktion (MWh)")

# add expected production from simulation
(pd.Series(pvgis).cumsum()/1e3).loc[production_monthly_sum.index].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet", ax=ax_n)

# bar plot of monthly production
(production_monthly_sum_cum/1e3).plot(marker="o", ax=ax_n, color=solar_color, alpha=0.7, lw = 2, zorder = 10)

ax_i = fig.add_subplot(gs[1,:])
ax_i.set_title("Daglig produktion (kWh)")
df.plot(ax= ax_i, lw = 0, alpha=0.6, color = solar_color)
ax_i.fill_between(df.index, 0, df["Produktion"], color=solar_color, alpha=0.3, zorder = 0)
ax_i.set_ylim(0, ax_i.get_ylim()[1])

# Layout
for ax in [ax_m, ax_n, ax_i]:
    # hide upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(lw = 0.5, ls='--', color='gray', alpha=0.7)
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_ylabel("")
    ax.set_xlabel("")

ax_i.legend().set_visible(False)

# add copyright on bottom right of the figure
fig.text(0.42, 0.01, '© 2025 Universitetets Energifællesskab (UEF)', ha='right', va='bottom', fontsize=12, color='gray', alpha=0.7)

# savefig
fig.savefig("figures/production_" + year_months[-1] + "_panel.png", bbox_inches='tight')
