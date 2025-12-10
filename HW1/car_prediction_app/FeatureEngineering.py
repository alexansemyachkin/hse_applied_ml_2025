import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()

        if "year" in df.columns:
            df["year2"] = df["year"] ** 2

        if "km_driven" in df.columns:
            df["log_km"] = np.log1p(df["km_driven"])

        if "max_power" in df.columns and "year" in df.columns:
            df["power_year"] = df["max_power"] * df["year"]

        if "max_power" in df.columns and "engine" in df.columns:
            df["hp_per_liter"] = df["max_power"] / df["engine"]

        if "owner" in df.columns:
            df["owner_3plus"] = df["owner"].isin(
                ["Third Owner", "Fourth & Above Owner"]
            ).astype(int)

        return df
