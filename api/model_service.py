import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DEFAULT_DATASET_PATH = os.environ.get(
    "DERMA_DATASET_PATH",
    r"C:\Users\user\Desktop\eng\2eme\Machine_Learnning\ML_project\dermatology_database_1.csv",
)
DEFAULT_MODEL_PATH = os.environ.get(
    "DERMA_MODEL_PATH",
    r"C:\Users\user\Desktop\eng\2eme\Machine_Learnning\ML_project\api\model_artifacts.joblib",
)

CLASS_NAME_MAP = {
    1: "Psoriasis",
    2: "Seborrheic dermatitis",
    3: "Lichen planus",
    4: "Pityriasis rosea",
    5: "Chronic dermatitis",
    6: "Pityriasis rubra pilaris",
}


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_names: List[str]
    class_names: List[str]
    model_name: str
    feature_stats: Dict[str, Dict[str, float]]


def _find_dataset_file(dataset_path: str) -> str:
    if os.path.isfile(dataset_path):
        return dataset_path
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            "Dataset path not found. Set DERMA_DATASET_PATH to a valid CSV or folder."
        )
    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the dataset directory.")
    return os.path.join(dataset_path, csv_files[0])


def _prepare_dataframe(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "class" not in df.columns:
        raise ValueError("Expected a 'class' column in the dataset.")

    X = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce")
    y = df["class"]
    feature_names = X.columns.tolist()
    return X, y, feature_names


def _build_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "SVM (RBF)": SVC(kernel="rbf", C=3, gamma="scale", probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
        )
    except Exception:
        pass

    return models


def _select_best_model(X: pd.DataFrame, y: pd.Series) -> ModelArtifacts:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for name, model in _build_models().items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        score = f1_score(y_test, preds, average="macro")
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = name

    if best_pipeline is None:
        raise RuntimeError("No model could be trained.")

    best_pipeline.fit(X, y_encoded)

    class_names = [
        CLASS_NAME_MAP.get(int(label), f"Class {label}") for label in encoder.classes_
    ]

    feature_stats = {
        col: {
            "min": float(np.nanmin(X[col])),
            "max": float(np.nanmax(X[col])),
            "mean": float(np.nanmean(X[col])),
        }
        for col in X.columns
    }

    return ModelArtifacts(
        pipeline=best_pipeline,
        feature_names=X.columns.tolist(),
        class_names=class_names,
        model_name=best_name,
        feature_stats=feature_stats,
    )


class ModelService:
    def __init__(self, dataset_path: str | None = None, model_path: str | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.artifacts = self._load_or_train(dataset_path)

    def _load_or_train(self, dataset_path: str | None) -> ModelArtifacts:
        if os.path.isfile(self.model_path):
            return joblib.load(self.model_path)

        csv_path = _find_dataset_file(dataset_path or DEFAULT_DATASET_PATH)
        X, y, _ = _prepare_dataframe(csv_path)
        artifacts = _select_best_model(X, y)
        joblib.dump(artifacts, self.model_path)
        return artifacts

    def metadata(self) -> Dict[str, object]:
        features = [
            {
                "name": name,
                "min": self.artifacts.feature_stats[name]["min"],
                "max": self.artifacts.feature_stats[name]["max"],
                "mean": self.artifacts.feature_stats[name]["mean"],
            }
            for name in self.artifacts.feature_names
        ]
        return {
            "features": features,
            "classes": self.artifacts.class_names,
            "model": self.artifacts.model_name,
        }

    def predict(self, features: Dict[str, float], top_n: int = 3) -> Dict[str, object]:
        missing = [name for name in self.artifacts.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")

        ordered_values = [features[name] for name in self.artifacts.feature_names]
        df = pd.DataFrame([ordered_values], columns=self.artifacts.feature_names)
        probas = self.artifacts.pipeline.predict_proba(df)[0]
        sorted_indices = np.argsort(probas)[::-1][:top_n]

        top = [
            {
                "label": self.artifacts.class_names[int(idx)],
                "probability": float(probas[int(idx)]),
            }
            for idx in sorted_indices
        ]
        alert = float(probas[sorted_indices[0]]) < 0.5

        return {
            "top": top,
            "alert": alert,
            "model": self.artifacts.model_name,
        }
