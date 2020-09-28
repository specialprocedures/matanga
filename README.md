# Online markets for illicit drugs in Georgia

## Replication code and source data

This repository contains the source data and replication code for the report "Online markets for illicit drugs in Georgia". The repository consists of the following files and folders:
- [scrape.py](https://github.com/crrcgeorgia/matanga/blob/master/scrape.py): the scraper for Matanga. This ceased working and was repaired periodically throughout the study as changes to Matanga broke the script. It has not been updated since ceasing functioning in mid-August 2020 and will require further adjustment to operate.
- [clean.py](https://github.com/crrcgeorgia/matanga/blob/master/clean.py): consolidation of data and post processing
- [sales.py](https://github.com/crrcgeorgia/matanga/blob/master/sales.py): sales estimation function, called from [clean.py](https://github.com/crrcgeorgia/matanga/blob/master/clean.py)
- [utils.py](https://github.com/crrcgeorgia/matanga/blob/master/utils.py): utility functions, used to avoid cluttering [clean.py](https://github.com/crrcgeorgia/matanga/blob/master/clean.py)
- [policy_brief.py](https://github.com/crrcgeorgia/matanga/blob/master/policy_brief.py): Anaylsis of data derived from the above.

This code is presented as was at the time of study completion and comes with no guarantees. If you are interested in replicating the analysis, please contact the author at [i.goodrich@crrccenters.org](mailto:i.goodrich@crrccenters.org).

Source data for the study is also provided in the following folders:

- [SCRAPED](https://github.com/crrcgeorgia/matanga/DATA/INPUT/SCRAPED): Source data drawn from [scrape.py](https://github.com/crrcgeorgia/matanga/blob/master/scrape.py)
- [LISTINGS](https://github.com/crrcgeorgia/matanga/DATA/INPUT/LISTINGS/CSV): All listing data following consolidation and cleaning. Observed at listing per scrape.
- [SALES](https://github.com/crrcgeorgia/matanga/DATA/INPUT/SALES/CSV): All sales data following sales estimation. Observed at listing per day.
