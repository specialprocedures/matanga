#%%
print("Importing modules")
import pandas as pd

import os
import datetime
from datetime import date, timedelta
from forex_python.bitcoin import BtcConverter
from forex_python.converter import CurrencyRates
from sales import caluclate_sales
from utils import time_stamp, consolidate_data, label_df, recategorise, add_cuts

#%%
TIME_STAMP = time_stamp()
SOURCE_PATH = "DATA/INPUT/SCRAPED"
SAVE_PATH = "DATA/OUTPUT"

#%%
print("Consolidating scraped data")
# Pulling raw scraped data from CSVs, I really need to get into SQL
df = consolidate_data(SOURCE_PATH, SAVE_PATH, TIME_STAMP)

# Consistent records only exist from Feb 4th
START_DATE = datetime.datetime(2020, 2, 4)
END_DATE = pd.to_datetime(datetime.date.today())

df = df[(df.time_stamp >= START_DATE) & (df.time_stamp < END_DATE)]

# Stripping out and reordering columns
df = df[
    [
        "source",
        "time_stamp",
        "region",
        "sell_code",
        "sell_name",
        "description",
        "desc_en",
        "готовых",
        "предзаказ",
        "вес",
        "$",
        "€",
        "BTC",
        "Saburtalo",
        "Vake",
        "Дигоми",
        "Old Tbilisi (Старый город)",
        "Varketili",
        "Gldani",
        "Nadzaladevi",
        "Didube",
        "Vazisubani",
    ]
]

# Google translate eat your heart out
col_translate = {
    "готовых": "ready",
    "предзаказ": "pending",
    "вес": "quantity",
    "Дигоми": "Dighomi",
    "Old Tbilisi (Старый город)": "Old Tbilisi",
    "$": "usd",
}

df = df.rename(columns=col_translate)

# Stripping out some unwanted сирилик
df[["ready", "pending"]] = df[["ready", "pending"]].replace("нет", 0).astype(int)
df["quantity"] = df["quantity"].str[:-2].astype(float)

# And set the day
df["date"] = pd.to_datetime(df["time_stamp"].dt.date)

#%% This is a very large and complex function, buried in utils
print("Labelling substances")
df = label_df(df)

#%%
# Pull in bitcoin prices and convert prices to USD.
print("Converting bitcoin prices")
b = BtcConverter()
rates = b.get_previous_price_list("USD", START_DATE, END_DATE)
btc = (
    pd.DataFrame(rates, index=[0])
    .T.reset_index()
    .rename(columns={"index": "date", 0: "btc_usd"})
)

btc["date"] = pd.to_datetime(btc["date"])

df = pd.merge_asof(
    df.sort_values("time_stamp"),
    btc.sort_values("date"),
    left_on="date",
    right_on="date",
)

df["usd_calc"] = df["btc_usd"] * df["BTC"]
df["usd"] = df["usd"].fillna(df["usd_calc"])


# Some weirdly don't give BTC price
c = CurrencyRates()
df_dates = df["date"].dt.date.unique()
eur_rates = [c.get_rate("EUR", "USD", i) for i in df_dates]

eur = pd.DataFrame([df_dates, eur_rates]).T.rename(columns={0: "date", 1: "eur_usd"})
eur["date"] = pd.to_datetime(eur["date"])
df = pd.merge_asof(
    df.sort_values("time_stamp"),
    eur.sort_values("date"),
    left_on="date",
    right_on="date",
)
df["usd_eur_calc"] = df["eur_usd"] * df["€"]
df["usd"] = df["usd"].fillna(df["usd_eur_calc"])


#%% recategorise based on existing list


#%% Slap in a uid for each record, based on a hash of key columns

print("Hashing records and fixing outliers")

HASH_COLS = [
    "sell_code",
    "region",
    "desc_en",
    "group",
    "sub",
    "type",
    "quantity",
]

df["uid"] = df[HASH_COLS].apply(tuple, axis=1).apply(hash)

#%% # Correct an error for some listings where some idiot put usd in the btc box

# Detect extreme outliers in USD price
error_list = df[df["usd"] > df["usd"].max() - (df["usd"].std() * 2)]["uid"].unique()

# Build a new set of usd values calculated the correct way
error_tab = df[df["uid"].isin(error_list)]
error_cor = error_tab["usd"] / error_tab["btc_usd"]

# And replace in the main df
df.loc[df.index.isin(error_cor.index), "usd"] = error_cor


#%% A couple of useful columns for later analysis
df["usd_ready"] = df["usd"] * df["ready"]
df["quantity_ready"] = df["quantity"] * df["ready"]
df["unit_price"] = df["usd"] / df["quantity"]


#%%


print("Exporting cleaned listings to csv and xlsx")
df.reset_index(inplace=True, drop=True)
# and out to Excel, as Excel has lousy uft-8 support for csvs, and I do my
# checking in Excel


df.to_excel(f"DATA/OUTPUT/LISTINGS/EXCEL/listings_{TIME_STAMP}.xlsx", index=False)
df.to_csv(f"DATA/OUTPUT/LISTINGS/CSV/listings_{TIME_STAMP}.csv", index=False)

#%%
print("Calculating sales")
sales = caluclate_sales(df=df, timestamp=TIME_STAMP)

print("Exporting sales to csv and xlsx")
sales.to_excel(f"DATA/OUTPUT/SALES/EXCEL/sales_{TIME_STAMP}.xlsx", index=False)
sales.to_csv(f"DATA/OUTPUT/SALES/CSV/sales_{TIME_STAMP}.csv", index=False)

# %% Building base stats

# Now building aggregated stats by substance per day

# Basic stats
agg_dict = {
    "sold": "sum",
    "sold_pending": "sum",
    "usd_sold": "sum",
    "usd_sold_pending": "sum",
    "uid": "nunique",
    "sell_code": "nunique",
    "quantity": "median",
}

# Apply agg_dict and rename columns
tab = (
    sales.groupby(["date", "sub", "type"])
    .agg(agg_dict)
    .rename(
        columns={
            "uid": "unique_listings",
            "sell_code": "unique_vendors",
            "sold": "ready_transactions",
            "sold_pending": "presale_transactions",
            "usd_sold": "ready_revenue",
            "usd_sold_pending": "presale_revenue",
            "quantity": "deal_size",
        }
    )
).reset_index()

#%%

# Add cut points and weekdays
tab["cut"] = add_cuts(tab["date"])
tab["weekday"] = tab["date"].dt.day_name()

# Make sure we don't include incomplete dates (i.e. today)
cut = pd.to_datetime(date.today() - timedelta(days=1))
tab = tab[tab["date"] <= cut]

# saving column order for later as it gets messed up by the next bit and I'd like to keep it
col_ord = tab.columns.to_list() + ["ppu_modal_quant"]

# We want the price per unit at the modal quantity, this gets a bit complex, bear with me.

# Helper function to pull the modal quantity from a grouped df
def get_top(x):
    return x.loc[x.sold == x.sold.max()]["quantity"]


#%%

# Build a table of modal quantities for all subs
modal_quantities = (
    sales.groupby(["sub", "type", "quantity"])["sold"]  # Group by
    .sum()  # and find total sales at each quant
    .reset_index()  # Let's not muck about with weird indices
    .groupby(["sub", "type"])  # Group again, but for substance/type only
    .apply(get_top)  # Apply our helper function
    .reset_index()  # Flatten us out again
    .drop("level_2", axis=1)  # Get rid of an unecessary column
    .drop_duplicates(
        subset=["sub", "type"]
    )  # For smaller subs, sometimes two modal quants. Take lowest.
)

#%%

# Mean ppu for all substances at all quantities
mean_ppu = (
    sales.groupby(["date", "sub", "type", "quantity"])["usd"].mean().reset_index()
)

# Get a list of tuples for the given keys out of our modal quantities table
# i.e. just the modal quantities
keys = ["sub", "type", "quantity"]
mod_idx = modal_quantities.set_index(keys).index


#%%

# Filter the modal quantities based on the above index
modal_price = (
    mean_ppu.set_index(keys)
    .loc[mod_idx]
    .reset_index()
    .set_index(["date", "sub", "type"])["usd"]
    .reset_index()
    .rename(columns={"usd": "ppu_modal_quant"})
)

# Merge back into the main tab
tab = tab.merge(modal_price, on=["date", "sub", "type"], how="left")[col_ord]
#%%
# Calculate totals
tab["total_transactions"] = tab["ready_transactions"] + tab["presale_transactions"]

tab["total_revenue"] = tab["ready_revenue"] + tab["presale_revenue"]

# listings["date"] = pd.to_datetime(listings["date"])

# Adding in total packages ready on any given day
# This is based on max() and may be lower than transactions
# due to carry over and restocking

# This wasn't used in the final analysis, an interim daily summary sheet. May not agree with policy_brief.py as a lot of
# complicated analysis goes on there and this was something of a prototype.


# tab = (
#     df.groupby(["uid", "date", "sub", "type"], observed=True)["ready"]
#     .max()
#     .reset_index()
#     .groupby(["date", "sub", "type"], observed=True)["ready"]
#     .sum()
#     .reset_index()
#     .rename(columns={"ready": "total_ready_packages"})
#     .merge(tab, on=["date", "sub", "type"], how="left")
# )
#%%

# tab.to_excel(f"DATA/OUTPUT/DAILY/daily_sub_stats_cut_{TIME_STAMP}.xlsx", index=False)


#%%
