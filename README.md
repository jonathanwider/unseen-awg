<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/wg_white_with_text.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/source/images/wg.png">
  <img alt="unseen-awg logo" src="docs/source/images/wg.png" width="500">
</picture>

# unseen-awg: Spatio-temporal weather generation using analogs and unseen data

*unseen-awg* provides a method for generating spatio-temporal weather data by resampling "UNSEEN" reforecast data, i.e. data from weather forecasts initialized with historical weather conditions.

**Want to analyze simulations created with the weather generator?** We provide 500 21-year simulations and a reforecast dataset of impact-relevant atmospheric variables over Europe at 0.4° resolution. See [ECMWF Data](#ecmwf-data).

**Want to generate new simulations with a weather generator?** You can use our provided weather generator with precomputed similarities.

**Want to set up your own (*unseen-awg*) weather generator?** We provide a reforecast dataset of preprocessed geopotential height fields as a starting point.

## Accompanying resources
|Resource|Link|
|---|---|
| Paper preprint | To be submitted. |
| User guide | [https://jonathanwider.github.io/unseen-awg/](https://jonathanwider.github.io/unseen-awg/) |
| Archived input ECMWF data ([overview](ecmwf-data)) | [https://www.wdc-climate.de/ui/entry?acronym=unsawg_inp](https://www.wdc-climate.de/ui/entry?acronym=unsawg_inp) |
| *unseen-awg* instance and simulations | [https://www.wdc-climate.de/ui/entry?acronym=unsawg_wg](https://www.wdc-climate.de/ui/entry?acronym=unsawg_wg) |
| Evaluation code | [https://codebase.helmholtz.cloud/jonathan.wider/eval-unseen-awg](https://codebase.helmholtz.cloud/jonathan.wider/eval-unseen-awg)|

## Installation
Installation instructions are specified in the [user guide](#accompanying-resources).

## Data & model availability
We release data to allow reproducing our results and creating new *unseen-awg* weather generators as described in the table of [accompanying resources](#accompanying-resources).
A high-level introduction on how to work with these data is given in the [user guide](#accompanying-resources).

## ECMWF data
Our work utilizes data from the European Centre for Medium-Range Weather Forecasts (ECMWF, [www.ecmwf.int](www.ecmwf.int)). 

In the [accompanying resources](#accompanying-resources), we make available a preprocessed version of their "Extended ensemble forecast hindcast" dataset [1] from cycle 48r1 of the Integrated Forecast System (IFS). The preprocessed version follows the input data format used by *unseen-awg*.

This data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0) license https://creativecommons.org/licenses/by/4.0/

Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.

Additionally, we include preprocessed daily ERA5 data retrieved from the Copernicus Climate Change Service [2, 3]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

In the `data/` subdirectory, we include a land-sea mask derived from ERA5 data (CC BY 4.0).

> [1] ECMWF (2023). IFS Documentation CY48R1 - Part V: Ensemble Prediction System. DOI: [https://doi.org/10.21957/E529074162](https://doi.org/10.21957/E529074162)

> [2] Hersbach, H., Comyn-Platt, E., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N., Cagnazo, C., Cucchi, M. (2023): ERA5 post-processed daily-statistics on pressure levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: [https://doi.org/10.24381/cds.4991cf48](https://doi.org/10.24381/cds.4991cf48)

> [3] Copernicus Climate Change Service, Climate Data Store, (2024): ERA5 post-processed daily-statistics on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: [https://doi.org/10.24381/cds.4991cf48](https://doi.org/10.24381/cds.4991cf48)



An example simulation created with *unseen-awg*:

<div align="center">
  <img src="docs/source/images/animation_all_vars.gif" width="600" alt="Gif of unseen-awg simulations showing fields of geopotential height at 500hPa, daily total precipitation sums, and daily mean, maximum, and minimum temperatures over Europe."/>
</div>