import pandas as pd
import os


def to_int_cpra(x):
    # 你可以改成 int(x) 或 round(x)
    return int(round(float(x)))

def DATA_BANK(path="./data/"):

    """ Path of the data"""
    pair_path = path + "pairs.csv" #./data/pairs.csv
    alts_path = path + "alts.csv" #./data/alts.csv

    """ Load the Pair Data"""
    pair_bank = []
    if not os.path.exists(pair_path):
        print("There is no files (pair data) at path ---> Using Default DataBank")
        pair_bank = [
        {
            "type": "pair",
            "candidate_abo": "B",
            "paired_donor_abo": "A",
            "cpra": 20,
            "wait_days": 60,
            "candidate_age": 45,
            "prior_living_donor": False,
            "orphan": False,
            "center": 1,
            "renege_prob": 0.02,

            # recipient (candidate) HLA antigens (A/B/DR), 2 each
            "candidate_hla_A": [2, 24],
            "candidate_hla_B": [7, 44],
            "candidate_hla_DR": [4, 15],

            # paired donor HLA antigens (A/B/DR), 2 each
            "paired_donor_hla_A": [1, 24],
            "paired_donor_hla_B": [8, 44],
            "paired_donor_hla_DR": [4, 11],
        },
        {
            "type": "pair",
            "candidate_abo": "O",
            "paired_donor_abo": "B",
            "cpra": 30,
            "wait_days": 200,
            "candidate_age": 30,
            "prior_living_donor": False,
            "orphan": False,
            "center": 2,
            "renege_prob": 0.02,

            "candidate_hla_A": [3, 11],
            "candidate_hla_B": [35, 51],
            "candidate_hla_DR": [1, 13],

            "paired_donor_hla_A": [3, 26],
            "paired_donor_hla_B": [35, 60],
            "paired_donor_hla_DR": [1, 4],
        },
        {
            "type": "pair",
            "candidate_abo": "AB",
            "paired_donor_abo": "O",
            "cpra": 10,
            "wait_days": 10,
            "candidate_age": 12,
            "prior_living_donor": False,
            "orphan": False,
            "center": 1,
            "renege_prob": 0.02,

            "candidate_hla_A": [1, 2],
            "candidate_hla_B": [7, 8],
            "candidate_hla_DR": [4, 11],

            "paired_donor_hla_A": [1, 2],
            "paired_donor_hla_B": [7, 27],
            "paired_donor_hla_DR": [4, 11],
        },]
    else:
        pairs = pd.read_csv(pair_path)
        for _, r in pairs.iterrows():
            pair_bank.append({
                "type": "pair",
                "candidate_abo": r["ABO_CAND"],
                "paired_donor_abo": r["ABO_DON"],
                "cpra": to_int_cpra(r["CPRA_AT_MATCH_RUN"]),
                "wait_days": 0,  # csv file do not provide these information
                "candidate_age": int(r["AGE_AT_ADD_CAND"]),
                "prior_living_donor": False,  # csv file do not provide these information
                "orphan": False,              # csv file do not provide these information
                "center": 0,                  # csv file do not provide these information
                "renege_prob": 0.02,

                "candidate_hla_A": [int(r["CA1"]), int(r["CA2"])],
                "candidate_hla_B": [int(r["CB1"]), int(r["CB2"])],
                "candidate_hla_DR": [int(r["CDR1"]), int(r["CDR2"])],

                "paired_donor_hla_A": [int(r["DA1"]), int(r["DA2"])],
                "paired_donor_hla_B": [int(r["DB1"]), int(r["DB2"])],
                "paired_donor_hla_DR": [int(r["DDR1"]), int(r["DDR2"])],
            })

    """ Load the Alts Data"""
    altruist_bank = []
    if not os.path.exists(alts_path):
        print("There is no files (alts data) at path ---> Using Default DataBank")
        altruist_bank = [
        {
            "type": "altruist",
            "donor_abo": "O",
            "center": 1,
            "donor_hla_A": [2, 24],
            "donor_hla_B": [7, 44],
            "donor_hla_DR": [4, 15],
        },
        {
            "type": "altruist",
            "donor_abo": "A",
            "center": 2,
            "donor_hla_A": [1, 3],
            "donor_hla_B": [8, 35],
            "donor_hla_DR": [1, 4],
        },
        {
            "type": "altruist",
            "donor_abo": "B",
            "center": 3,
            "donor_hla_A": [11, 26],
            "donor_hla_B": [51, 60],
            "donor_hla_DR": [13, 15],
        },]

    else:
        alts = pd.read_csv(alts_path)
        for _, r in alts.iterrows():
            altruist_bank.append({
                "type": "altruist",
                "donor_abo": r["ABO_DON"],
                "center": 0,  # csv file do not provide these information
                "donor_hla_A": [int(r["DA1"]), int(r["DA2"])],
                "donor_hla_B": [int(r["DB1"]), int(r["DB2"])],
                "donor_hla_DR": [int(r["DDR1"]), int(r["DDR2"])],
            })
    return pair_bank, altruist_bank
