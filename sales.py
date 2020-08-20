import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def cum_diff(x):
    vec = x["ready"].diff()
    mask = x["ready"].shift(1) > x["ready"]
    return abs(vec.mul(mask).sum())


def cum_diff_pend(x):
    vec = x["pending"].diff()
    mask = x["pending"].shift(1) > x["pending"]
    return abs(vec.mul(mask).sum())


def caluclate_sales(df=None, timestamp=None, path=None):

    ALL_COLS = [
        "time_stamp",
        "uid",
        "date",
        "sell_code",
        "region",
        "desc_en",
        "description",
        "ready",
        "pending",
        "quantity",
        "group",
        "sub",
        "type",
        "usd",
    ]

    GROUP_COLS = [
        "sell_code",
        "region",
        "desc_en",
        "group",
        "sub",
        "type",
        "quantity",
    ]

    df_original = df.copy()

    # Drop uneccesary listings
    df = df[~df.group.isin(["Work/Reagent/Other", "Combo"])]

    # Trim the date to size. I just want the one month.
    START_DATE = datetime(2020, 2, 4)
    END_DATE = datetime.now()
    df = df[(df.time_stamp >= START_DATE) & (df.time_stamp <= END_DATE)].sort_values(
        "time_stamp"
    )

    """ One of the big challenges with sales estimation is getting sales per day.
        This is problematic because when the estimation function is applied per day
        any changes between days are lost.

        For example, if the last scrape of the day was at 10:00pm with 6 ready
        and the next scrape was at 9:00am the next day with 4 ready, the two sales
        between scrapes would be ignored.

        This section brings forward the last value of each day to the beginning of
        the next to avoid this problem.
        """

    # Create a df with dates per uid
    m = df[["date", "uid"]].drop_duplicates().rename(columns={"date": "time_stamp"})
    df = pd.concat([df, m], sort=False).sort_values(["uid", "time_stamp"])

    # Set empty dates to next day
    df["date"] = df.groupby("uid")["date"].bfill()

    # Bring forward values from previous day
    df[[i for i in df if i != "uid"]] = df.groupby("uid").ffill()

    df = df.dropna(subset=GROUP_COLS)

    """ This function is the heart of the sales estimation process. It works by
        creating two vectors:

        a cumulative difference vector e.g:
            f([6, 5, 7, 2, 1]) = [nan, 1, -2, 5, 1]

        and a logical vector to spot upwards changes:
            f([6, 5, 7, 2, 1]) = [False, True, False, True, True]

        As False evaluates to 0, the two can be multiplied leaving only decreases
            [nan, 1, -2, 5, 1] * [False, True, False, True, True] = [nan, 1, 0, 5, 1]

        This vector can then be summed, giving the total sales:
            sum([nan, 1, 0, 5, 1]) = 7

        """

    # We first group by date, uid and GROUP_COLS (which may be duplication
    # but belt and braces) and then apply the cummulative difference function.

    # What this does is split the df into loads of smaller dfs, based on each listing
    # to which the function is applied.

    # Following the process, we need to add 'usd' (price) back in, because price is
    # volatile and thus can't be used for grouping. I take the _minimum_ from a day
    # because for most days for most listings, all prices will be the same. Where
    # There is a difference, I use the minimum to show that on this day, the price
    # changed.
    print('Applying sales calculation on ready packages')
    sold = (
        df.groupby(["date", "uid"] + GROUP_COLS, observed=True)
        .apply(cum_diff)
        .reset_index()
        .merge(df.groupby(["date", "uid"])["usd"].min().reset_index())
        .rename(columns={0: "sold"})
    )

    sold["quantity_sold"] = sold["sold"] * sold["quantity"]
    sold["usd_sold"] = sold["sold"] * sold["usd"]
    sold["unit_price"] = sold["usd"] / sold["quantity"]

    print('Applying sales calculation on preorders')
    sold_pending = (
        df.groupby(["date", "uid"] + GROUP_COLS, observed=True)
        .apply(cum_diff_pend)
        .reset_index()
        .merge(df.groupby(["date", "uid"])["usd"].min().reset_index())
        .rename(columns={0: "sold_pending"})
    )

    sold_pending["quantity_sold_pending"] = (
        sold_pending["sold_pending"] * sold_pending["quantity"]
    )
    sold_pending["usd_sold_pending"] = (
        sold_pending["sold_pending"] * sold_pending["usd"]
    )

    print('Merging and applying scrape count')
    out = sold.merge(sold_pending, on=["date"] + GROUP_COLS, how="outer")

    # Add in scrape_count to adjust for over/underscraping
    scrape_count = (
        df_original.groupby("date")["source"]
        .nunique()
        .reset_index()
        .rename(columns={"source": "scrape_count"})
    )

    out = out.merge(scrape_count, on="date", how="left")

    # and out to Excel, as Excel has lousy uft-8 support for csvs, and I do my
    # checking in Excel

    out = out.drop([i for i in out if "_y" in i], axis=1).rename(
        columns={i: i.split("_")[0] for i in out if "_x" in i}
    )

    
    # out.to_excel(f"DATA/OUTPUT/SALES/EXCEL/sales_{timestamp}.xlsx", index=False)
    # out.to_csv(f"DATA/OUTPUT/SALES/CSV/sales_{timestamp}.csv", index=False)
    # out.to_feather(f"DATA/OUTPUT/SALES/FEATHER/sales_{timestamp}.feather")

    return out
