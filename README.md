# UEF-PV-Production

This repo uses GitHub Actions to create monthly reports of the production from the [UEF](https://www.uef.dk/home) solar PV installation. 

It first fetches data reported at the [AURORA dashboard](https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=2025-06) and saves it in the `data/` folder.

Subsequently, the workflow creates two graphs that summarize the production within the past months (with daily and monthly resolution, respectively) and saves them in the `figures/` folder.

The figures are then attached to an automatically generated email which is shared with members of UEF (note that all sensitive data is encrypted and contained as secret variables).

The workflow is run on the first day in every month.

![UEF PV installation production data](/figures/monthly_production_2025-05.png)
