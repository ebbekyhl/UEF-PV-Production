---
layout: default
title: UEF PV Production
---

# Summary of energy yield

This page summarizes the energy yield from the [UEF](https://www.uef.dk/home) solar PV installation in Aarhus, Denmark.

Panel overview:

- [Panel 1]({{ site.baseurl }}/panel-1/)
- [Panel 2]({{ site.baseurl }}/panel-2/)
- [Panel 3]({{ site.baseurl }}/panel-3/)

The summary is produced daily via Github Actions, and it currently acquires data from the following sources:
- Reported energy yield from [AURORA dashboard](https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=2025-06)
- Inverter data from [martavp/UEF](https://github.com/martavp/UEF)
- Electricity prices from [Energi Data Service](https://www.energidataservice.dk/)
- Grid CO2 intensity from [Energi Data Service](https://www.energidataservice.dk/)

The energy yields are compared with simulated estimates from [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/). 

The workflow includes a CO2 accounting of the emissions associated with manufacturing the panels, balanced by the avoided emissions from providing CO2-free electricity to the consumer. Here, avoided emissions are calculated based on the CO2-intensity of the electricity that would otherwise have been imported from the local grid. 

The GitHub action workflow includes an email sent to recipients listed in the secret variable "EMAIL_TO" with the created figures attached to.

Data is updated daily, while emails are sent either monthly or quarterly.

![UEF PV installation production data](/figures/production_panel_1.png)

![UEF PV installation production data](/figures/production_panel_2.png)

![UEF PV installation production data](/figures/production_panel_3.png)
