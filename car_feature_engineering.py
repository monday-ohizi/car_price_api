# feature_engineering.py
import pandas as pd

def add_features(df):
    df = df.copy()
    df["Make_Model"] = df["Make"] + "_" + df["Model"]
    df["Is_Automatic"] = df["Transmission"].str.contains("Auto", case=False, na=False).astype(int)
    top_fuels = df["Fuel Type"].value_counts().nlargest(3).index
    df["Fuel_Type_Simple"] = df["Fuel Type"].where(df["Fuel Type"].isin(top_fuels), "Other")
    df["Car_Age"] = 2025 - df["Year"]
    return df
