import download
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import os

download_dir = download_dir = os.path.abspath("data")  # choose where to save data

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
        "Maj": 12132.2,
        "Jun": 12686.7,
        "Jul": 11868.2,
        "Aug": 9501.8,
        "Sep": 6703.4,
        "Okt": 3242.3,
        "Nov": 1108.7,
        "Dec": 562.9}

# For now, assume the following self-consumptions:
self_consumption_ratio = pd.Series({"Jan": 0.85,
                          "Feb": 0.85,
                          "Mar": 0.85,
                          "Apr": 0.85,
                            "Maj": 0.85,
                            "Jun": 0.85,
                            "Jul": 0.85,
                            "Aug": 0.85,
                            "Sep": 0.85,
                            "Okt": 0.85,
                            "Nov": 0.85,
                            "Dec": 0.85
                            })

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
df.rename(columns={"Ep":"Produktion"}, inplace=True)

# convert "date" into pd.datetime
df["date"] = pd.to_datetime(df["date"])

# set "date" as index
df.set_index("date", inplace=True, drop=True)

# df comes in daily values, here we convert it to monthly values
production_monthly_sum = df.groupby(df.index.month).sum() # sum of production per month
production_monthly_mean = df.groupby(df.index.month).mean() # mean of production per month

production_monthly_sum.index = production_monthly_sum.index.map(month_mapping)
production_monthly_sum.columns = ["Produktion " + year_months[-1][0:4]]

production_monthly_sum_cum = production_monthly_sum.cumsum()

production_last_month = production_monthly_sum.iloc[-1].item()  # get the last month production values

def format_number(number):
    formatted_number = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted_number

def extract_month_name(year_month):
    return month_mapping_long[int(year_month.split("-")[1])]

def extract_year(year_month):
    return year_month.split("-")[0]

with open("email_summary.txt", "w", newline="") as f:
    f.write(f"Samlet produktion for {extract_month_name(year_months[-1])} {extract_year(year_months[-1])}: {format_number(production_last_month)} kWh\r\n")

########################################################################################
########################### Read grid data #############################################
########################################################################################
start = '2025-01-01'
today = pd.Timestamp.now()
end = str(today.year) + "-" + str(today.month).zfill(2) + "-" + (str(today.day - 6).zfill(2))
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

net_tariffs = 7.63 + 9.25 + 22.60 # Transmissionsnettarif + Systemtarif + Nettarif (all in cents/kWh)
# https://energinet.dk/el/elmarkedet/tariffer/aktuelle-tariffer/
g_prices["ElPriceDKK"] = g_prices["SpotPriceDKK"] + net_tariffs / 100

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
################################## Panel B #############################################
########################################################################################

solar_color = "#f9a202"
blue_color = "#1f77b4"
red_color = "#b93020"

aspect_ratio = 11.69 / 8.27
fig0, ax = plt.subplots(figsize=(10*aspect_ratio,10), nrows = 3, sharex=False)

shares = df_emissions_carrier.sum()/(df_emissions_carrier.sum().sum())*100

df_emissions_carrier = df_emissions_carrier[shares[shares > 0.1].index]

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
ax[2].annotate("Negativ spotpris", 
               xy=(pd.to_datetime("7/7/2025"), -0.55),
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
ax[0].set_title("Daglig Elproduktion i DK1 [GWh]")
ax[1].set_title(r"Daglig CO$_2$ intensitet i DK1 [gCO$_2$/kWh]")
ax[2].set_title("Spotpris + Nettarif (ekskl. elafgift) i DK1 [DKK/kWh]")

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
fig0.savefig("figures/production_panelB.png", bbox_inches='tight')

########################################################################################
################################## Panel A #############################################
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
ax_m.set_title("Månedlige produktionsværdier (MWh)")

# bar plot of monthly production
(production_monthly_sum/1e3).plot(kind="bar", ax=ax_m, color=solar_color, alpha=0.7, edgecolor="k", width=0.7)
self_consumption = production_monthly_sum["Produktion " + year_months[-1][0:4]]*self_consumption_ratio.loc[production_monthly_sum.index]
(self_consumption/1e3).plot(kind="bar", ax=ax_m, color=solar_color, alpha=0.4, 
                            edgecolor="k", 
                            hatch="/",
                            width=0.7)

# add expected production from simulation
(pd.Series(pvgis)/1e3).loc[production_monthly_sum.index].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet", ax=ax_m)

ax_n = fig.add_subplot(gs[0,1])
ax_n.set_title("Kumuleret produktion (MWh)")

# add expected production from simulation
(pd.Series(pvgis).cumsum()/1e3).loc[production_monthly_sum.index].plot(marker="X", ls="--", color="k", alpha=0.6, label="Forventet produktion", ax=ax_n)

# cumulative sum of monthly production
(production_monthly_sum_cum/1e3).plot(marker="o", ax=ax_n, color=solar_color, alpha=0.7, lw = 2, zorder = 10)
(self_consumption.cumsum()/1e3).plot(ls="-", marker="o", ax=ax_n, color="gray", alpha=0.7, lw = 2, zorder = 5, label = "Egetforbrug " + year_months[-1][0:4])

# Daily production
ax_i = fig.add_subplot(gs[1,:])
ax_i.set_title("Daglig produktion (kWh)")
df.plot(ax= ax_i, lw = 0, alpha=1, color = solar_color)
ax_i.fill_between(df.index, 0, df["Produktion"], color=solar_color, alpha=0.3, zorder = 0)
ax_i.set_ylim(0, ax_i.get_ylim()[1])

# Avoided CO2 emissions
ax_k = fig.add_subplot(gs[2,:])
ax_k.set_title(r"Kumuleret potentielt undgået CO$_2$ (tCO$_2$)") # CO2 udledninger undgået ved egenproduktion
CO2_avoided = df_emissions_intensity * df["Produktion"]  
(CO2_avoided.cumsum()/1e6).plot(ax=ax_k, color='green', lw=1, alpha=0.7, label="Undgået CO2")
ax_k.fill_between(CO2_avoided.cumsum().index, 0, CO2_avoided.cumsum()/1e6, color='lightgreen', alpha=0.3, zorder = 0)
ax_k.set_ylim(0, ax_k.get_ylim()[1])

# Layout 
for ax in [ax_m, ax_n, ax_i, ax_k]:
    # hide upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(lw = 0.5, ls='--', color='gray', alpha=0.7)
    ax.legend(loc = "best")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_ylabel("")
    ax.set_xlabel("")
    if ax != ax_m and ax != ax_n:
        ax.set_xlim([df.index.min(), df.index.max()])

ax_m.legend().set_visible(False)
ax_i.legend().set_visible(False)
ax_k.legend().set_visible(False)

# add hatch to legend in ax_m
handles, labels = ax_m.get_legend_handles_labels()
hatch_handle = plt.Rectangle((0,0),1,1, facecolor="white", edgecolor="k", hatch="//", alpha=0.4)
handles.append(hatch_handle)
labels.append("Egetforbrug")
ax_m.legend(handles[-1:], labels[-1:], bbox_to_anchor=(0.02, 0.98), loc='upper left', borderaxespad=0.)

# add copyright on bottom right of the figure
fig.text(0.42, 0.02, '© 2025 Universitetets Energifællesskab (UEF)', ha='right', va='bottom', fontsize=14, color='gray', alpha=0.7)
fig.savefig("figures/production_panelA.png", bbox_inches='tight')

########################################################################################
############################# Save panel A and B #######################################
########################################################################################
pngs = ["figures/production_panelA.png", 
        "figures/production_panelB.png"]

# --- Make a PDF with those PNGs as pages ---
with PdfPages("figures/UEF_rapport.pdf") as pdf:
    for path in pngs:
        img = mpimg.imread(path)
        h, w = img.shape[:2]

        # Create a figure sized to the image (1:1 at 100 dpi = w/100 by h/100 inches)
        dpi = 100.0
        fig_PDF = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
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
