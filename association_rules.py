import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

ARM_RULES = None
FREQ_ITEMS = None
TRAINED = False


def prepare_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        items = []
        # Stream
        if pd.notna(row.get("A_stream")):
            items.append(f"stream={row['A_stream']}")
        # City tier
        if pd.notna(row.get("A_city_tier")):
            items.append(f"tier={row['A_city_tier']}")
        # Education
        if pd.notna(row.get("A_education_level")):
            items.append(f"edu={row['A_education_level']}")
        # Career domains
        for d in ["E_career_domain_1","E_career_domain_2","E_career_domain_3"]:
            if pd.notna(row.get(d)):
                items.append(f"domain={row[d]}")
        # Learning preference
        if pd.notna(row.get("E_learning_preference")):
            items.append(f"learn={row['E_learning_preference']}")
        # WTP band
        if pd.notna(row.get("F_wtp_monthly_band")):
            items.append(f"wtp={row['F_wtp_monthly_band']}")
        # Platform adoption
        if pd.notna(row.get("F_will_use_platform")):
            items.append(f"will_use={row['F_will_use_platform']}")
        # Biggest challenge
        if pd.notna(row.get("C_biggest_challenge")):
            items.append(f"challenge={row['C_biggest_challenge']}")
        # Persona
        if pd.notna(row.get("persona_label")):
            items.append(f"persona={row['persona_label']}")
        # Feature preferences (high = 4+)
        for feat in ["E_feat_roadmap","E_feat_career_match","E_feat_scholarship","E_feat_job_market"]:
            if pd.notna(row.get(feat)) and row.get(feat, 0) >= 4:
                items.append(f"wants_{feat.replace('E_feat_','')}")
        # Family decision
        if pd.notna(row.get("C_family_decision_authority")):
            items.append(f"family={row['C_family_decision_authority']}")
        # Pricing preference
        if pd.notna(row.get("F_pricing_model_pref")):
            items.append(f"pricing={row['F_pricing_model_pref']}")

        if items:
            transactions.append(items)
    return transactions


def run_association_rules(df, min_support=0.05, min_confidence=0.5, min_lift=1.2):
    global ARM_RULES, FREQ_ITEMS, TRAINED

    transactions = prepare_transactions(df)

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_te = pd.DataFrame(te_array, columns=te.columns_)

    freq_items = apriori(df_te, min_support=min_support, use_colnames=True, max_len=4)
    FREQ_ITEMS = freq_items

    if len(freq_items) == 0:
        return pd.DataFrame()

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules["support"] = rules["support"].round(4)
    rules["confidence"] = rules["confidence"].round(4)
    rules["lift"] = rules["lift"].round(4)

    ARM_RULES = rules
    TRAINED = True
    return rules


def get_top_rules(n=15):
    if ARM_RULES is None or len(ARM_RULES) == 0:
        return pd.DataFrame()
    return ARM_RULES.head(n)[["antecedents_str","consequents_str","support","confidence","lift"]]


def get_business_interpretation(rules_df):
    interpretations = []
    for _, row in rules_df.head(5).iterrows():
        ant = row.get("antecedents_str","")
        cons = row.get("consequents_str","")
        lift = row.get("lift", 1)
        conf = row.get("confidence", 0)
        interp = f"Students with [{ant}] are {lift:.1f}x more likely to have [{cons}] (confidence: {conf*100:.0f}%)"
        interpretations.append(interp)
    return interpretations
