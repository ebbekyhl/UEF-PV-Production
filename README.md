# UEF-PV-Production

This repo uses GitHub Actions to create monthly reports of the production from the [UEF](https://www.uef.dk/home) solar PV installation. 

It first fetches data reported at the [AURORA dashboard](https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=2025-06) and saves it in the `data/` folder.

Subsequently, the workflow creates two graphs that summarize the production within the past months (with daily and monthly resolution, respectively) and saves them in the `figures/` folder.

The figure with monthly resolution is attached to an automatically generated email and shared with recipients listed in the secret variable "EMAIL_TO".

The workflow is run once every month (currently set to run on the 5th, to account for delays in the data reporting on the AURORA dashboard).

![UEF PV installation production data](/figures/monthly_production_2025-07.png)
