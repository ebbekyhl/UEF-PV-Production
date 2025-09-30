# UEF-PV-Production

This repo uses GitHub Actions to create monthly or quarterly reports of the production from the [UEF](https://www.uef.dk/home) solar PV installation. 

It acquires data from the [AURORA dashboard](https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=2025-06) and saves it in the `data/` folder.

Graphs summarizing the production within the past months (with daily and monthly resolution, respectively) are saved in the `figures/` folder.

We compare production data with estimated values from [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/). 

The created figure is attached to an automatically generated email and shared with recipients listed in the secret variable "EMAIL_TO".

The workflow is run once every month (currently set to run on the 5th, to account for delays in the data reporting on the AURORA dashboard) or quarterly.

![UEF PV installation production data](/figures/production_2025-08_panel.png)
