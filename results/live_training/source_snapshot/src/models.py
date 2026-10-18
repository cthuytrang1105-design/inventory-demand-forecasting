"""Ba thuật toán ML và mốc seasonal naive cùng đặc trưng, cùng protocol."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_processing import KEYS
from src.feature_engineering import NUMERIC


def build_model(name, params=None, quick=False):
    if name == "SeasonalNaive":
        return None
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=60 if quick else 160,
                                               max_depth=12, min_samples_leaf=2,
                                               n_jobs=1, random_state=42),
        # Native sklearn boosting giữ bộ cài gọn, không cần libomp/LightGBM ngoài.
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=70 if quick else 180,
                                                              max_leaf_nodes=15, learning_rate=0.08,
                                                              early_stopping=False, random_state=42),
    }
    if name not in models:
        raise ValueError(f"Mô hình không hỗ trợ: {name}")
    prep = ColumnTransformer([
        ("number", StandardScaler(), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="error", sparse_output=False), KEYS),
    ])
    pipe = Pipeline([("preprocess", prep), ("model", models[name])])
    if params:
        pipe.set_params(**params)
    return pipe
