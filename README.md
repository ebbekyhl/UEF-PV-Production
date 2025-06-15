# UEF-PV-Production

This repo uses GitHub Actions to create monthly reports of the production from the UEF solar PV installation. 

It first fetches data reported at the AURORA dashboard (https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=2025-06) and saves it in the 'data/' folder.

Subsequently, the workflow creates two graphs that summarize the production within the past months and saves them in the 'figures/' folder.

The workflow is run on the first day in every month.
