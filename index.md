---
layout: default
title: UEF PV Production
---

# Summary

This page describes the energy yield from the [UEF](https://www.uef.dk/home) solar PV installation in Aarhus, Denmark.

Here’s a short podcast episode produced with Google NotebookLM explaining the data below:

<audio controls>
  <source src="{{ 'Denmark_s_Solar_Challenge__Citizen_Cooperatives.m4a' | relative_url }}" type="audio/mp4">
  Your browser does not support the audio element.
</audio>

## Data Overview

Here is the visualization and description of the data…

Figure panels overview:

- [Panel 1: Intraday power supply]({{ site.baseurl }}/panel-1/)
- [Panel 2: Energy yield and CO2 footprint]({{ site.baseurl }}/panel-2/)
- [Panel 3: Energy mix, CO2 intensity, and electricity prices of the local grid]({{ site.baseurl }}/panel-3/)
- [Panel 4: Average electricity prices and PV yield intraday curves]({{ site.baseurl }}/panel-4/)

The information shown on this site is updated using Github Actions. The workflow currently acquires data from the following sources:
- Reported energy yield from [AURORA dashboard](https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=2025-06)
- Inverter data from [martavp/UEF](https://github.com/martavp/UEF)
- Electricity prices from [Energi Data Service](https://www.energidataservice.dk/)
- Grid CO2 intensity from [Energi Data Service](https://www.energidataservice.dk/)

Furthermore, the energy yields are compared with simulated estimates from [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/). 

The workflow includes a CO2 accounting of the emissions associated with manufacturing the panels, balanced by the avoided emissions from providing CO2-free electricity to the consumer. Here, avoided emissions are calculated based on the CO2-intensity of the electricity that would otherwise have been imported from the local grid. 

The visualizations and data presented on this page are updated daily, while some data sources might be refreshed less frequently. 

![UEF PV installation production data](/figures/production_panel_1.png)

![UEF PV installation production data](/figures/production_panel_2.png)

![UEF PV installation production data](/figures/production_panel_3.png)

![UEF PV installation production data](/figures/production_panel_4.png)
