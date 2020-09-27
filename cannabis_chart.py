#%%
import pandas as pd
from datetime import date
import seaborn as sns
import math
from utils import add_cuts, pill_strength

import statsmodels.api as sm
import statsmodels.stats.api as sms

sns.set_style("dark")
colors = ["#44546a", "#2baae9", "#ff575d", "#00375e", "#a5a5a5"]
my_pal = sns.color_palette(colors)
sns.set_palette(my_pal)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
from matplotlib import ticker
from pandas.plotting import register_matplotlib_converters
from matplotlib import gridspec
import statsmodels.formula.api as smf

register_matplotlib_converters()


#%%

# Import sales and listings
sales = pd.read_excel("DATA/OUTPUT/SALES/EXCEL/sales_20200817_1240.xlsx")
listings = pd.read_excel(
    "DATA/OUTPUT/LISTINGS/EXCEL/listings_20200817_1240.xlsx", sheet_name="Sheet1"
)
daily = pd.read_excel("DATA/OUTPUT/DAILY/daily_sub_stats_cut_20200817_1240.xlsx")
time_stamp = "20200817_1240"

# Unit price for sales
sales["unit_price"] = sales["usd"] / sales["quantity"]

# Dealing with a miscategorised methadone transaction
mis_meth = sales[
    (sales["sub"] == "Methadone")
    & (sales["type"] == "Powder")
    & (sales["unit_price"] < 100)
]["uid"].unique()
sales.loc[sales["uid"].isin(mis_meth), "type"] = "Syrup"
listings.loc[listings["uid"].isin(mis_meth), "type"] = "Syrup"

# Add cut points
listings["cut"] = add_cuts(listings["date"])
sales["cut"] = add_cuts(sales["date"])

# Add pre-post Cov cut
listings["cov_cut"] = listings["date"] >= pd.to_datetime(date(2020, 2, 26))
sales["cov_cut"] = sales["date"] >= pd.to_datetime(date(2020, 2, 26))

# Ensuring only complete dates present
max_date = pd.to_datetime(date(2020, 8, 16))
listings = listings[listings["date"] <= max_date]
sales = sales[sales["date"] <= max_date]
daily = daily[daily["date"] <= max_date]

min_date = pd.to_datetime(date(2020, 2, 5))
listings = listings[listings["date"] >= min_date]
sales = sales[sales["date"] >= min_date]
daily = daily[daily["date"] >= min_date]

#%%
cann = sales.groupby(['date', 'sub'])['usd_sold'].sum().reset_index().query('sub == "Cannabis"').drop('sub', axis=1)
not_cann = sales.groupby(['date', 'sub'])['usd_sold'].sum().reset_index().query('sub != "Cannabis"').groupby('date')['usd_sold'].sum().reset_index().assign(sub='Other substances')

cann_df = pd.concat([cann, not_cann]).set_index('date').groupby('sub').rolling(7)['usd_sold'].mean().reset_index()


#%%
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=cann_df, x="date", y="usd_sold", hue="sub")

# Set x and y limits
# ax.set_ylim((0, 100000))
ax.set_xlim(date(2020, 2, 1), date(2020, 8, 17))

# Set date locating and formating for the xasis
locator = mdates.AutoDateLocator(minticks=5, maxticks=14)
formatter = mdates.ConciseDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Dropping uncessary legend title and putting legend entries in order
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:]
labels = ["Methadone powder", "Other opiates"]

# Locating the legend
ax.legend(
    loc="upper center", ncol=2, handles=handles, labels=labels, facecolor="white",
)

# Labelling and formatting axes
ax.set_ylabel("Number of listings")
ax.set_xlabel("")

# ax.get_yaxis().set_major_formatter(
#     ticker.FuncFormatter(lambda x, p: format(int(x), ","))
# )

# The title
fig.suptitle(
    "Unique opiate listings (February-August, 2020)\nSeven-day rolling average",
    y=1.1,
    x=0.52,
    fontsize=20,
)

# Tidying and saving/showing
fig.tight_layout()
fig.savefig("CHARTS/BRIEF/0817/opiates.png", bbox_inches="tight")
plt.show()
