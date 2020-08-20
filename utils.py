import datetime
import os
import pandas as pd
from datetime import date, timedelta
from statsmodels.stats.weightstats import DescrStatsW
import re


def time_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def consolidate_data(
    path: str, save_path: str, time_stamp: str, feather=True, excel=True, csv=False
):

    # Pull in files from raw csvs
    out = []
    for file in [f"{path}/" + f for f in os.listdir(path) if "csv" in f]:
        frame = pd.read_csv(file)

        if "Tbilisi" not in frame["region"].unique():
            continue
        frame["source"] = file
        out.append(frame)

    # Concatenate and ensure the timestamp is set correctly
    df = pd.concat(out, sort=False).reset_index(drop=True)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], dayfirst=True)

    # Export to desired format(s)
    # if feather:
    #     df.to_feather(f"{save_path}/RAW/FEATHER/raw_{time_stamp}.feather")
    if excel:
        df.to_excel(f"{save_path}/RAW/EXCEL/raw_{time_stamp}.xlsx", index=False)
    if csv:
        df.to_csv(f"{save_path}/RAW/CSV/raw_{time_stamp}.csv", index=False)

    # Return the data
    return df


# Absurdly long rules based system for labelling. Great puzzle thinking how
# to do this properly, but this is good enough for now. Well worth a refactor
# if I ever get time.


def get_all_labs(x, first_time=False):

    processed = x.lower().strip()

    out_dict = {}
    labels = pd.read_csv("DATA/INPUT/labels.csv")
    out = []  # Out list, later reduced to len(1) through rules-based algo
    # Run through all of the labels
    for row in labels.itertuples():
        group = row[1]  # Group from labels
        subgroup = row[2]  # Subgroup from labels
        labs = row[3]  # A comma delim string w. labels e.g ("cannabis, weed")
        for lab in [i.strip() for i in labs.split(",")]:  # Split the string
            if "&" in lab:  # Some require two strings (e.g. bio AND green)
                if (
                    lab.split("&")[0].strip() in processed  # Need both 4 match
                    and lab.split("&")[1].strip() in processed
                ):
                    out.append((group, subgroup))
                    continue

            elif lab == "sk":
                if lab in processed.split():
                    out.append((group, subgroup))

            else:
                if lab in processed:  # Check for presence of string
                    out.append((group, subgroup))

                elif " ".join(list(lab.strip())) in x.lower():  # s p a c e d
                    out.append((group, subgroup))

    out = list(set(out))  # Remove duplicated
    out_dict["list"] = out
    out_dict["string"] = x
    return out_dict


def parse_labs(out_dict: dict):
    labs = out_dict["list"]
    desc = out_dict["string"]

    if len(labs) == 1:  # len() == 1 is great
        return labs[0]

    if len(labs) == 0:  # len() == 0 means "no clue"
        return ("No matches", "No matches")

    if len(labs) >= 2:  # Some combinations will generally return 2
        for autohit in ["Work", "Reagent", "Combo"]:
            if any(autohit in lab for lab in labs):
                return (autohit, autohit)

        for sub, typ in [
            ("Cannabis", "Resin"),
            ("Methadone", "Syrup"),
            ("MDMA", "Pill"),
        ]:

            if any(sub in lab for lab in labs) and any(typ in lab for lab in labs):
                return (sub, typ)

        if any("Methamphetamine" in lab for lab in labs) and any(
            "Amphetamine" in lab for lab in labs
        ):
            return ("Methamphetamine", "Methamphetamine")

        # Weed has loads of weird names, sometimes referencing other drugs
        if any("Cannabis" in lab for lab in labs):
            return ("Cannabis", "Cannabis")

        else:  # Non-matches to the uncategorised bin
            if "+" in desc:
                return ("Combo", "Combo")

            else:
                return ("Uncategorised", "Uncategorised")


def label_from_desc(x: str):
    out = get_all_labs(x)
    return parse_labs(out)


def label_df(df):
    # The actual labelling process
    df["tup"] = df["desc_en"].apply(lambda x: label_from_desc(x))

    df["sub"], df["type"] = [
        pd.Categorical(df["tup"].apply(lambda x: x[n])) for n in [0, 1]
    ]
    df.drop("tup", axis=1, inplace=True)

    # Some further grouping. This should really be in labels, but life is short
    groups = {
        "Opiates/Opiate Substitutes": [
            "Methadone",
            "Subutex",
            "Buprenorphine",
            "Heroin",
        ],
        "NPS": ["Mephedrone", "25i-NBOMe", "Spice", "A-PVP", "MPDV", "Unknown"],
        "Pharmaceuticals": ["Gabapentin", "Bupropion Hydrochloride"],
        "Pychedelics": ["LSD", "DMT"],
        "Amphetamines": ["Amphetamine", "Methamphetamine"],
        "Work/Reagent/Other": ["Work", "Reagent", "Check"],
    }

    # Setting the grouping dict around the right way
    group2 = {}
    for k, v in groups.items():
        for vx in v:
            group2.update({vx: k})

    # Applying grouping labels
    df["group"] = pd.Categorical(
        df["sub"].apply(lambda x: group2[x] if x in group2 else x)
    )
    return df


def recategorise(df: pd.DataFrame, path="DATA/OUTPUT/CATEGORIES/uncategorised_r.xlsx"):

    recats = pd.read_excel(path).drop_duplicates(subset="desc_en")

    for col in ["sub", "group", "type"]:
        df[col] = df[col].replace({"Uncategorised": None})
        d = {k: v for k, v in zip(recats["desc_en"], recats[col])}
        df[col] = df[col].fillna(df["desc_en"].replace(d))

    return df


def weighted(x):
    stats = DescrStatsW(x["quantity"], x["sold"])
    return {"median": stats.quantile(0.5)[0.5], "std": stats.std}


# Working out the means and medians at each of the +/- 1std quantities
def best_price(x, stats):
    sub = x["sub"].iloc[0]
    type = x["type"].iloc[0]
    med, std = stats.loc[(sub, type)]
    y = x[(x.quantity >= med - std) & (x.quantity <= med + std)]["unit_price"]

    return {"median_ppu_norm": y.median(), "mean_ppu_norm": y.mean()}


# Function to apply the cutpoints


def add_cuts(vec, path="DATA/INPUT/short_dates.csv"):
    dts = pd.read_csv(path)["date"]
    dts = pd.to_datetime(dts)

    lbs = pd.read_csv(path)["policy"]

    bins = [vec.min()] + dts.to_list() + [vec.max() + timedelta(days=2)]

    labs = ["Pre-Covid"] + [i.split("\n")[0].strip() for i in lbs]

    out = pd.cut(vec, bins=bins, labels=labs, right=True, include_lowest=True)

    return out


def pill_strength(df):
    if df["sub"] == "MDMA" and df["type"] == "Pill":
        strength = re.findall("(\d{3})\s?[mg|MG|m.g|m g]", df["desc_en"])
        if len(strength) == 0:
            return None
        else:
            return max([int(i) for i in strength])
    else:
        return None
