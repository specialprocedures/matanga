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

# I only uploaded the csv files in the end, you'll need to change this bit and be careful about
# dtypes if you want to replicate the analysis, or alternatively re-run clean.py beforehand.

sales = pd.read_excel("DATA/OUTPUT/SALES/EXCEL/sales_20200817_1240.xlsx")
listings = pd.read_excel(
    "DATA/OUTPUT/LISTINGS/EXCEL/listings_20200817_1240.xlsx", sheet_name="Sheet1"
)


# daily = pd.read_excel("DATA/OUTPUT/DAILY/daily_sub_stats_cut_20200817_1240.xlsx")
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
# Convenience function to identify ready or pre-order listings
def ready_pre(row):
    if (row["ready"] == 0) and (row["pending"] > 0):
        return "Pre-order only"
    elif (row["pending"] == 0) and (row["ready"] > 0):
        return "Collection only"
    else:
        return "Both"


drop = ["Work", "Reagent", "Combo", "Services"]

# Applying to listings only
listings["ready_pre"] = listings.apply(ready_pre, axis=1)

# Fix the boths that move between ready and pre-order

# First make dict to map
ready_pre_dict = (
    listings.groupby("uid")["ready_pre"]
    .nunique()
    .reset_index()
    .query("ready_pre > 1")
    .assign(ready_pre="Both")
    .set_index("uid")
    .to_dict()["ready_pre"]
)

# Then apply conditionally
listings["ready_pre"] = listings.apply(
    lambda x: ready_pre_dict[x["uid"]]
    if x["uid"] in ready_pre_dict.keys()
    else x["ready_pre"], axis=1
)


# Similar for checking substance vs. non-substance
listings["list_cat"] = listings["sub"].apply(
    lambda x: "Substance" if x not in drop else "Non-Substance"
)

# Pivot table for unique/count by substance/non-substance, ready/preorder

listings_pivot = listings.pivot_table(
    index="ready_pre",
    columns="list_cat",
    values="uid",
    aggfunc=["nunique", "count"],
    margins=True,
)

listings_pivot.to_clipboard()

#%% sales-scrape bias regression for methods

Y = sales["sold"]
x = sales["scrape_count"]
X = sm.add_constant(x)

model = sm.OLS(Y, X)
results = model.fit()
results.summary()

#%% Identifying instances of restock
listings["restock"] = listings.groupby(["uid", "date"])["ready"].apply(
    lambda x: x > x.shift(1)
)
uid_restock = (
    listings.query('ready_pre == "Collection only"')
    .groupby(["uid", "date"])["restock"]
    .sum()
    .reset_index()
    .groupby(["uid"])
    .agg({"date": "nunique", "restock": "sum"})
    .reset_index()
)

# percentage non-restocks (and more)
uid_restock["restock"].value_counts(normalize=True)


#%%

# Drop non-substance and uncategorised listings

sales = sales[~sales["sub"].isin(drop)]
listings = listings[~listings["sub"].isin(drop)]

# Get pill strength
sales["pill_strength"] = sales.apply(pill_strength, axis=1).fillna("Not given")
listings["pill_strength"] = listings.apply(pill_strength, axis=1).fillna("Not given")

# Drop zero listings and zero sales records, but keep a backup first
sales_full = sales
listings_full = listings

listings = listings[listings["ready"] > 0]
sales = sales[sales["sold"] > 0]

#%% scraping count

dt_index = pd.date_range(sales["date"].min(), sales["date"].max())
fig, ax = plt.subplots(figsize=(12, 6))

scrapes_per_day = (
    pd.DataFrame(dt_index, columns=["date"])
    .merge(listings_full.groupby("date")["source"].nunique().reset_index(), how="left")
    .fillna(0)
)

hist_df = (
    scrapes_per_day["source"]
    .value_counts()
    .reset_index()
    .merge(pd.DataFrame(range(0, 13), columns=["index"]), how="right")
    .fillna(0)
    .sort_values(by="index")
)
hist_df["index"] = hist_df["index"].astype(int)

# Plotting
sns.barplot(data=hist_df, x="index", y="source", color=my_pal[0])
ax.set_ylabel("")
ax.set_xlabel("")
for _, row in hist_df.iterrows():
    y = row["source"]
    x = row["index"]

    ax.text(
        x=x, y=y + 2, s=str(int(y)), ha="center", va="center",
    )

fig.suptitle("Daily scrape counts (frequency)", fontsize=18)
plt.show()
fig.savefig("CHARTS/BRIEF/0817/scrape_hist.png", bbox_inches="tight")
#%% More bits for the methods paper

#%%

# Number of unique days
unique_days = listings_full["date"].nunique()

# Number of unique vendors
unique_vendors = listings_full["sell_code"].nunique()

# Number of unique listings
unique_listings = listings_full["uid"].nunique()

# Unique listings per city
listings_per_city = (
    listings_full.groupby("region")["uid"].nunique().div(unique_listings).mul(100)
)

# Total sales from the sum of usd_sold
total_sales = sales["usd_sold"].sum()

# Sales per day
average_day = total_sales / unique_days

# Ready/pre-order diff

ready_pre_tab = listings_full.groupby("uid")[["ready", "pending"]].min().reset_index()

ready_pre_percents = (
    ready_pre_tab.apply(ready_pre, axis=1).value_counts(normalize=True).mul(100)
)

ready_pre_sales = sales[["usd_sold", "usd_sold_pending"]].sum()


listings_full["usd_pending"] = listings_full["usd"] * listings_full["pending"]

# Listings contains multiple records for each listing, this gives the
# average usd_ready per day per listing, then groups by date and sums
# To give how much was listed as ready for collection each day
daily_listings = (
    listings_full.groupby(["date", "uid"])[["usd_ready", "usd_pending"]]
    .mean()
    .reset_index()
    .groupby("date")[["usd_ready", "usd_pending"]]
    .sum()
    .reset_index()
)

# Pre/post covid daily listings
daily_listings["cov_cut"] = daily_listings["date"] >= pd.to_datetime(date(2020, 2, 26))
cov_daily_average_listings = daily_listings.groupby("cov_cut")["usd_ready"].mean()

# Average usd value of all ready listings, based on the above
daily_listed_average = daily_listings.mean()

# Total usd sales per day
daily_sales = sales.groupby(["date"])["usd_sold"].sum().reset_index()

# Pre-post covid daily sales
daily_sales["cov_cut"] = daily_sales["date"] >= pd.to_datetime(date(2020, 2, 26))
cov_daily_average_sales = daily_sales.groupby("cov_cut")["usd_sold"].mean()

# Total transactions per substance
total_transactions_sub = daily.groupby("sub")["ready_transactions"].sum()

# Total transactions
total_transactions = total_transactions_sub.sum()

# Total transactions per substance as a percentage
total_transactions_sub_pct = total_transactions_sub / total_transactions


#%% Figure 2: Unique vendors and listings on the Matanga platform over time

# Get unique listings and vendors for each date and region
listings_tab = (
    listings.groupby(["date", "region"])[["uid", "sell_code"]].nunique().reset_index()
)

# Stick the regions an order I like using the Categorical function
listings_tab["region"] = pd.Categorical(
    listings_tab["region"], ["Tbilisi", "Kutaisi", "Batumi"]
)

# The actual plotting

# Configure layout
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(6, 1)

# Top plot
ax1 = fig.add_subplot(gs[0:2])
sns.lineplot(data=listings_tab, x="date", y="sell_code", style="region", ax=ax1)

ax1.get_xaxis().set_visible(False)  # hide axis label
ax1.get_legend().remove()  # hide legend
ax1.set_title(
    "Number of unique vendors (collection and pre-order)", fontsize=14
)  # Set title
ax1.set_ylim((0, 50))  # Set maximum y value
ax1.set_ylabel("")  # Hide y axis label
ax1.set_yticks([0, 25, 50])  # Specify my y ticks

# Bottom plot
ax2 = fig.add_subplot(gs[2:])
sns.lineplot(data=listings_tab, x="date", y="uid", style="region", ax=ax2)

ax2.set_title(
    "Number of unique listings (collection and pre-order)", fontsize=14
)  # Title
ax2.set_xlabel("")  # Hiding axis labels
ax2.set_ylabel("")  # Hiding axis labels
ax2.set_ylim((0, 150))  # Set maximum y value

# Bit fiddly, but all I'm doing is hiding the word "legend" from the legend
handles, labels = ax2.get_legend_handles_labels()
handles = handles[1:]
labels = labels[1:]

# And positioning the legend
ax2.legend(bbox_to_anchor=(0.666, -0.1), ncol=4, handles=handles, labels=labels)

# Telling matplotlib how I'd like the dates on the x axis
locator = mdates.AutoDateLocator(minticks=5, maxticks=14)
formatter = mdates.ConciseDateFormatter(locator)

# Looping through the two plots to sort a few things out
for ax in [ax1, ax2]:
    ax.xaxis.set_major_locator(locator)  # Setting dates on xaxis
    ax.xaxis.set_major_formatter(formatter)  # Formating dates on xaxis
    ax.tick_params(axis="both", which="major", labelsize=12)  # tidying

    # Throwing in the lines for Fridays
    for da in listings_tab["date"]:
        if da.day_name() == "Friday":
            ax.axvline(
                da, ymin=0, ymax=1, color="b", alpha=0.05, dashes=(5, 2, 1, 2),
            )

# Putting the title up top
fig.suptitle(
    "Matanga market dynamics (February-August, 2020)", y=1.05, x=0.52, fontsize=20,
)

# No-one like spines on their plots, except like chemistsy Phds or something
sns.despine(bottom=True, left=True)

# Tidying
fig.tight_layout()

# Rendering for mine eyes
plt.show()

# Saving for yours
fig.savefig("CHARTS/BRIEF/0817/unique_vend_list_time.png", bbox_inches="tight")

#%% Figure 3: Value of sales and listings on Matanga over time

# Total usd sales per substance per day
sales_tab = sales.groupby(["date", "sub"])["usd_sold"].sum().reset_index()

# Total value of ready listings per day
# As each listing appears multiple times per day in the listings data
# due to repeated scraping, I take the largest value each day (i.e.
# the most ready packages per listing)
ready_tab = (
    listings.groupby(["date", "sub", "uid"])["usd_ready"]  # Group by listing/sub/day
    .max()  # Take max, see above
    .reset_index()
    .groupby(["date", "sub"])["usd_ready"]  # Group by substance and day
    .sum()  # Get total daily available listings per substance
    .reset_index()
)

sr_tab = sales_tab.merge(ready_tab)  # Merge the two

# Get aggregate figures (not per substance) and melt to suit seaborn
sr_tot = (
    sr_tab.groupby("date")[["usd_sold", "usd_ready"]].sum().reset_index().melt("date")
)

# The actual ploting

# Set up plot
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=sr_tot, x="date", y="value", hue="variable")

# Set x and y limits
ax.set_ylim((0, 100000))
ax.set_xlim(date(2020, 2, 1), date(2020, 8, 17))

# Set date locating and formating for the xasis
locator = mdates.AutoDateLocator(minticks=5, maxticks=14)
formatter = mdates.ConciseDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Friday lines like the earlier chart
for da in sr_tot["date"]:
    if da.day_name() == "Friday":
        plt.axvline(
            da, ymin=0, ymax=1, color="b", alpha=0.05, dashes=(5, 2, 1, 2),
        )

# Dropping uncessary legend title and putting legend entries in order
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:][::-1]
labels = ["Revenue", "Listed"][::-1]

# Locating the legend
ax.legend(
    loc="upper center", ncol=2, handles=handles, labels=labels, facecolor="white",
)

# Throwing a little label at the bottom
ax.set_xlabel(
    "Note: Daily USD value of estimated sales and daily USD value of listings (collection only)",
    fontsize=10,
    x=0,
    horizontalalignment="left",
)

# Labelling and formatting axes
ax.set_ylabel("USD")
ax.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

# The title
fig.suptitle(
    "Total revenue and listings value (February-August, 2020)",
    y=1.05,
    x=0.52,
    fontsize=20,
)

# Adding covid outbreak
plt.axvline(date(2020,2,26), alpha=0.75, linestyle='--')
plt.text(x=date(2020,2,29), y=85000, s='First registered Covid-19 case\nFebruary 26, 2020')
# Tidying and saving/showing
fig.tight_layout()
fig.savefig("CHARTS/BRIEF/0817/list_sold_usd_time.png", bbox_inches="tight")
plt.show()


#%%

sales_sub = (
    sales.groupby(["sub", "type"])[["sold", "usd_sold"]]
    .sum()
    .sort_values(by="sold", ascending=False)
)

sorter = sales_sub.groupby("sub")["sold"].sum().sort_values(ascending=False)[
    :-10
].index.to_list() + ["Other"]


list_sub = listings_full.groupby(["sub", "type"])["uid"].nunique()

sales_sub = (
    sales_sub.merge(list_sub, left_index=True, right_index=True)
    .rename(columns={"uid": "u_list"})
    .reset_index()
)

for col in sales_sub.columns[2:]:
    sales_sub[col + "_pct"] = sales_sub[col].div(sales_sub[col].sum()).mul(100)

sales_sub_melt = (
    sales_sub.groupby("sub")[
        ["sold", "usd_sold", "u_list", "sold_pct", "usd_sold_pct", "u_list_pct"]
    ]
    .sum()
    .reset_index()
    .melt("sub")
)


def othering(df, cats, sorter):
    query = cats[0]
    out = (
        df.replace(
            df.query(f'variable == "{query}"')
            .sort_values(by="value")[:10]["sub"]
            .to_list()
            + ["Unknown"],
            "Other",
        )
        .groupby(["sub", "variable"])
        .sum()
        .reset_index()
    )

    out["sub"] = pd.Categorical(out["sub"])
    out["sub"].cat.set_categories(sorter, inplace=True)

    out["variable"] = pd.Categorical(out["variable"])
    out["variable"].cat.set_categories(cats, inplace=True)

    out = out.sort_values(by=["sub", "variable"]).dropna()

    return out


sales_sub_pct = othering(
    sales_sub_melt, ["sold_pct", "usd_sold_pct", "u_list_pct"], sorter
)
sales_sub_num = othering(sales_sub_melt, ["sold", "usd_sold", "u_list"], sorter)
#%% Plotting per substance sales and listings
sns.set(font_scale=1.5)
g = sns.catplot(
    data=sales_sub_pct,
    y="sub",
    x="value",
    col="variable",
    sharex=False,
    kind="bar",
    height=14,
    aspect=0.3875,
)

axes = g.axes.flatten()
titles = [
    "Number of transactions (#)\n",
    "Revenue (USD)\n",
    "Unique listings (#)\n",
]

col_tots = [sales_sub[i].sum() for i in ["sold", "usd_sold", "u_list"]]

for ax, t, n, row_tot in zip(axes, titles, [0, 1, 2], col_tots):
    ax.set_title(t)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    total_val = sum([i.get_width() for i in ax.patches])
    for p in ax.patches:

        num_val = int(p.get_width())
        text = f"  {int(num_val * row_tot / 100):,} | {int(num_val)}%"

        ax.text(
            p.get_width(), p.get_y() + p.get_height() / 2, text, ha="left", va="center",
        )

        p.set_facecolor(my_pal[n])
    ax.set_xlim(0, 100)

g.fig.suptitle("Per substance sales and listings", y=1.05, x=0.5, fontsize="28")
g.fig.tight_layout()
g.fig.savefig("CHARTS/BRIEF/0817/substances_other.png", bbox_inches="tight")

#%%
sub_stats = sales_sub.set_index(["sub", "type"]).merge(
    sales.groupby(["sub", "type"])["quantity_sold"].sum(),
    left_index=True,
    right_index=True,
)

modal_quants = (
    sales.groupby(["sub", "type", "quantity"], observed=True)["sold", "usd"]
    .agg({"sold": "sum", "usd": "median"})
    .sort_values("sold", ascending=False)
    .sort_index(level=["sub", "type"], sort_remaining=False)
    .reset_index()
    .groupby(["sub", "type"])
    .first()
    .drop("sold", axis=1)
    .rename(columns={"quantity": "modal_quant", "usd": "modal_quant_price"})
)

sub_stats = sub_stats.merge(modal_quants, left_index=True, right_index=True)
sub_stats.to_clipboard()

#%% Cannabis over time

cann = sales.groupby(['date', 'sub'])['usd_sold'].sum().reset_index().query('sub == "Cannabis"').drop('sub', axis=1)
not_cann = sales.groupby(['date', 'sub'])['usd_sold'].sum().reset_index().query('sub != "Cannabis"').groupby('date')['usd_sold'].sum().reset_index().assign(sub='Other substances')

cann_df = pd.concat([cann, not_cann])

sns.lineplot(data = cann_df, x='date', y='usd_sold', hue='sub')


#%%

# For this one and this one alone, I use the df without 0 sales dropped (sales_full)
sales_size = (
    sales_full.groupby(["sub", "type", "quantity"])["sold"]
    .sum()
    .reset_index(name="sold")
)
daily_avg_list = listings_full.groupby(["sub", "type", "quantity"])["uid"].nunique()

quant_tab = sales_size.merge(daily_avg_list, on=["sub", "type", "quantity"])
sns.set_style("dark")


def quant_plot(quant_tab, tup, unit="grams"):
    data = quant_tab.set_index(["sub", "type"]).loc[tup].reset_index()
    last_valid = data["uid"].last_valid_index()
    data = data.iloc[: last_valid + 1]

    data["cut"] = data["cut"].astype(str)
    data[["uid", "sold"]] = data[["uid", "sold"]].fillna(0)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    axes = ["", ""]
    axes[0] = plt.subplot(gs[0])
    axes[1] = plt.subplot(gs[1])

    heat = sns.heatmap(
        data=data[["uid"]].T,
        cmap=sns.color_palette("Blues", desat=0.3),
        annot=True,
        fmt="g",
        annot_kws={"fontsize": 10},
        ax=axes[1],
        cbar=False,
    )
    xlabs = data["cut"].unique()
    heat.set_xticklabels(xlabs, rotation=45, fontsize=12)
    heat.set_yticklabels(["Unique listings\n (inc. pre-order)"], rotation=0, fontsize=14)
    heat.set_xlabel(xlabel=f"Quantity ({unit})", fontsize=14)

    bar = sns.barplot(data=data, y="sold", x="cut", ax=axes[0], color=my_pal[0])
    bar.set(xlabel="", xticks=[])
    bar.set_ylabel(ylabel="Number of transactions", fontsize=14)
    for p in axes[0].patches:
        ylim = axes[0].get_ylim()[1]
        off_set = ylim / 20
        axes[0].annotate(
            "{:,.0f}".format(p.get_height()),
            (p.get_x() + (p.get_width() / 2), p.get_height() + off_set),  # Placement
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )

    y_up = math.ceil((ylim + (ylim / 5)) / 100) * 100

    axes[0].set_ylim((0, y_up))

    sns.despine(top=True, bottom=True, left=True, right=True)
    plt.subplots_adjust(hspace=0.1)

    if tup[0] == tup[1]:
        if tup[0] == "Cannabis":
            title = "Cannabis (herbal)"
        else:
            title = tup[0]

    else:
        title = f"{tup[0]} ({tup[1].lower()})"

    fig.suptitle(title, fontsize=24)

    path_stem = "_".join([i.lower() for i in tup])
    fig.savefig(f"CHARTS/BRIEF/0817/{path_stem}_quant.png", bbox_inches="tight")


#%%
quant_tab["cut"] = pd.cut(
    quant_tab["quantity"],
    bins=[0, 0.5, 1, 5, 10, 50, 100, 500, 1000, 10000],
    labels=[
        "0-0.5",
        "> 0.5-1",
        "> 1-5",
        "> 5-10",
        "> 10-50",
        "> 50-100",
        "> 100-500",
        "> 500-1000",
        "> 1000-10000",
    ],
)
quant_tab_cut = (
    quant_tab.groupby(["sub", "type", "cut"])[["sold", "uid"]].sum().reset_index()
)
quant_tab_cut[["sold", "uid"]] = quant_tab_cut[["sold", "uid"]]
tups = [
    ("Cannabis", "Cannabis"),
    ("Cannabis", "Resin"),
    ("MDMA", "Powder"),
    ("MDMA", "Pill"),
    ("Cocaine", "Cocaine"),
    ("Methadone", "Powder"),
    ("25i-NBOMe", "25i-NBOMe"),
    ("A-PVP", "A-PVP"),
]
units = ["grams", "grams", "grams", "pills", "grams", "grams", "blotters", "grams"]

for tup, unit in zip(tups, units):
    quant_plot(quant_tab_cut, tup, unit=unit)

#%%
sales_full.query('(sub == "MDMA") & (type == "Pill")').groupby("pill_strength")[
    ["uid", "usd_sold"]
].agg({"uid": "nunique", "usd_sold": "sum"}).to_clipboard()
#%%
sales_quant = sales.groupby(["sub", "type", "quantity"])["sold"].sum().reset_index()
sales_quant["rank"] = sales_quant.groupby(["sub", "type"])["sold"].rank(
    method="max", ascending=False
)

sales_quant = (
    sales_quant[sales_quant["rank"] == 1]
    .merge(
        sales.groupby(["sub", "type", "quantity"])["usd"].median(),
        how="left",
        on=["sub", "type", "quantity"],
    )
    .drop(["rank", "sold"], axis=1)
)

sales_quant = sales_quant.merge(
    sales.groupby(["sub", "type"])[["sold", "quantity_sold", "usd_sold"]]
    .sum()
    .reset_index(),
    on=["sub", "type"],
)
sales_quant.to_csv("DATA/OUTPUT/STATS/sub_stats.csv")

#%%
listings_full[
    listings_full["sub"] + listings_full["type"] == "MethadonePowder"
].groupby(["uid", "date"])[["ready", "pending"]].max().groupby(["date"])[
    ["ready", "pending"]
].sum().plot()

#%%

sub_quant_ready_sold = (
    listings_full.groupby(["uid", "date", "sub", "type", "quantity"])["ready"]
    .max()
    .groupby(["sub", "type", "quantity"])
    .sum()
    .reset_index()
    .merge(sales_size)
    .melt(["sub", "type", "quantity"])
)

#%%

#%% MDMA analysis

strength_tab["unique_listings"].sum()
listings_full.drop_duplicates(subset="uid").query("(sub=='MDMA') & (type=='Pill')")[
    "pill_strength"
].fillna(0).value_counts().sum()

#%% Methadone/opiate analysis
sales[sales["group"] == "Opiates/Opiate Substitutes"].groupby(["sub", "type"])[
    ["sold", "usd_sold"]
].sum().to_clipboard()


#%%

# Daily stats for opiates
op_day = (
    listings_full.query('group == "Opiates/Opiate Substitutes"')
    .groupby(["date", "uid", "sub", "type"], dropna=False)["ready"]
    .max()
    .groupby(["date", "sub", "type"], dropna=False)
    .sum()
    .reset_index()
    .merge(
        sales.query('group == "Opiates/Opiate Substitutes"')
        .groupby(["date", "sub", "type"], dropna=False)["sold"]
        .sum()
        .reset_index(),
        on=["date", "sub", "type"],
        how="outer",
    )
    .fillna(0)
)
op_day.to_clipboard()
#%%

op_day["group"] = None
op_day.loc[op_day["type"] == "Powder", "group"] = "Methodone Powder"
op_day["group"] = op_day["group"].fillna("Other opiates")

op_day = (
    op_day.groupby(["date", "group"])["ready", "sold"]
    .sum()
    .reset_index()
    .set_index("date")
    .groupby("group")
    .rolling(7)["ready", "sold"]
    .mean()
    .reset_index()
)
op_day.to_clipboard()
#%%
my_pal = sns.color_palette(colors)
sns.set_palette(my_pal)
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=op_day, x="date", y="ready", hue="group")

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
fig.savefig("CHARTS/BRIEF/0817/cannabis.png", bbox_inches="tight")
plt.show()


#%%
fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()

a = ax.plot(op_day["date"], op_day["ready"], label="Ready packages")
b = ax2.plot(op_day["date"], op_day["usd_sold"], dashes=(5, 2, 1, 2), label="Revenue")


# ax.set_yticklabels(list(range(0, 300)[::2]))
ax.set_ylim((0, 80))
ax.set_ylabel("Listed packages")
ax2.set_ylim((0, 5000))
ax2.set_ylabel("Revenue (USD)")
leg = a + b
labs = [l.get_label() for l in leg]
ax.legend(leg, labs, loc=0)

locator = mdates.AutoDateLocator(minticks=5, maxticks=14)
formatter = mdates.ConciseDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
fig.suptitle(
    "Methadone powder: market dynamics over time\nReady packages and revenue (USD) | 7-day rolling average"
)
fig.savefig("CHARTS/BRIEF/0817/methadone_time.png")

#%%

