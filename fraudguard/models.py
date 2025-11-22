from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_logreg_model(preprocessor) -> Pipeline:
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return model


def build_forest_model(preprocessor) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return model
