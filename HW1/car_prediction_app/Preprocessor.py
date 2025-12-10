import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):

    def parse_rpm(self, x):
        if isinstance(x, str) and "-" in x:
            a, b = x.split("-")
            a = float(a.replace(",", "").replace(" ", ""))
            b = float(b.replace(",", "").replace(" ", ""))
            return (a + b) / 2
        try:
            return float(str(x).replace(",", "").replace(" ", "")) if x != "" else np.nan
        except:
            return np.nan

    def parse_torque_column(self, df):
        df = df.copy()

        torque_raw = df["torque"].astype(str)

        df["torque_value"] = torque_raw.str.extract(r"([0-9]*\.?[0-9]+)").astype(float)
        df["torque_unit"] = torque_raw.str.extract(r"(nm|NM|Nm|kgm|KGm|kgm?)")

        df["torque"] = np.where(
            df["torque_unit"].str.lower().str.contains("kg", na=False),
            df["torque_value"] * 9.8,
            df["torque_value"]
        )

        rpm_extract = torque_raw.str.extract(
            r"([0-9][0-9,.-]*[0-9])(?=[^0-9]*rpm)",
        )

        df["rpm_raw"] = rpm_extract[0].fillna("")
        df["max_torque_rpm"] = df["rpm_raw"].apply(self.parse_rpm)

        df = df.drop(columns=["torque_value", "torque_unit", "rpm_raw"])

        return df

    def clean_numeric(self, col):
        return (
            col.astype(str)
               .str.extract(r"([0-9]*\.?[0-9]+)")
               .astype(float)
        )

    def fix_dataset(self, df):
        df = df.copy()

        for c in ["mileage", "engine", "max_power", "seats"]:
            if c in df.columns:
                df[c] = self.clean_numeric(df[c])

        if "torque" in df.columns:
            df = self.parse_torque_column(df)

        if "name" in df.columns:
            df["name"] = df["name"].str.split().str[0]

        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df = self.fix_dataset(df)

        return df
