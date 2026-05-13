import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import davies_bouldin_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Get paths relative to this file
_CURRENT_DIR = Path(__file__).parent
_PROJECT_ROOT = _CURRENT_DIR.parent

_DATASET_FALLBACK_API = _CURRENT_DIR / "dermatology_database_1.csv"
_DATASET_FALLBACK_ROOT = _PROJECT_ROOT / "dermatology_database_1.csv"

DEFAULT_DATASET_PATH = os.environ.get(
    "DERMA_DATASET_PATH",
    str(_DATASET_FALLBACK_API if _DATASET_FALLBACK_API.exists() else _DATASET_FALLBACK_ROOT),
)

_CLASSIFIER_FALLBACK = _CURRENT_DIR / "model_artifacts.joblib"
_CLASSIFIER_DEFAULT = _CURRENT_DIR / "classification_artifacts.joblib"

DEFAULT_CLASSIFIER_PATH = os.environ.get(
    "DERMA_CLASSIFIER_PATH",
    os.environ.get(
        "DERMA_MODEL_PATH",
        str(_CLASSIFIER_FALLBACK if _CLASSIFIER_FALLBACK.exists() else _CLASSIFIER_DEFAULT),
    ),
)
DEFAULT_SEGMENTATION_PATH = os.environ.get(
    "DERMA_CLUSTER_PATH",
    str(_CURRENT_DIR / "segmentation_artifacts.joblib"),
)
DEFAULT_RECOMMENDATION_PATH = os.environ.get(
    "DERMA_RECO_PATH",
    str(_CURRENT_DIR / "recommendation_artifacts.joblib"),
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
class ClassificationArtifacts:
    pipeline: Pipeline
    feature_names: List[str]
    class_names: List[str]
    model_name: str
    feature_stats: Dict[str, Dict[str, float]]
    f1_macro: float


@dataclass
class SegmentationArtifacts:
    imputer: SimpleImputer
    scaler: StandardScaler
    model: KMeans
    model_name: str
    k: int
    metrics: Dict[str, float]
    feature_names: List[str]
    feature_stats: Dict[str, Dict[str, float]]
    cluster_sizes: Dict[int, int]
    benchmark: List[Dict[str, object]]


@dataclass
class RecommendationArtifacts:
    pipeline: Pipeline
    feature_names: List[str]
    class_names: List[str]
    model_name: str
    feature_stats: Dict[str, Dict[str, float]]
    top_features: List[Dict[str, float]]


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


def _compute_feature_stats(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        col: {
            "min": float(np.nanmin(X[col])),
            "max": float(np.nanmax(X[col])),
            "mean": float(np.nanmean(X[col])),
        }
        for col in X.columns
    }


def _encode_labels(y: pd.Series) -> Tuple[LabelEncoder, np.ndarray, List[str]]:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = [
        CLASS_NAME_MAP.get(int(label), f"Class {label}") for label in encoder.classes_
    ]
    return encoder, y_encoded, class_names


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


def _train_classification(X: pd.DataFrame, y: pd.Series) -> ClassificationArtifacts:
    _, y_encoded, class_names = _encode_labels(y)

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
    feature_stats = _compute_feature_stats(X)

    return ClassificationArtifacts(
        pipeline=best_pipeline,
        feature_names=X.columns.tolist(),
        class_names=class_names,
        model_name=best_name,
        feature_stats=feature_stats,
        f1_macro=float(best_score),
    )


def _train_segmentation(X: pd.DataFrame) -> SegmentationArtifacts:
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    benchmark: List[Dict[str, object]] = []

    best_k = 2
    best_sil = -1.0
    best_model: KMeans | None = None
    best_dbi = 0.0

    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        dbi = float(davies_bouldin_score(X_scaled, labels))
        benchmark.append(
            {
                "name": f"KMeans (K={k})",
                "clusters": int(k),
                "silhouette": sil,
                "davies_bouldin": dbi,
                "noise": 0,
            }
        )
        if sil > best_sil:
            best_sil = sil
            best_model = model
            best_k = k
            best_dbi = dbi

    eps_values = [0.5, 0.7, 1.0, 1.3, 1.6]
    min_samples_values = [5, 10, 20]
    for eps in eps_values:
        for ms in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=ms)
            labels = dbscan.fit_predict(X_scaled)
            cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = int((labels == -1).sum())
            sil: float | None = None
            dbi: float | None = None
            if cluster_count >= 2:
                mask = labels != -1
                if mask.sum() > cluster_count:
                    sil = float(silhouette_score(X_scaled[mask], labels[mask]))
                    dbi = float(davies_bouldin_score(X_scaled[mask], labels[mask]))
            benchmark.append(
                {
                    "name": f"DBSCAN (eps={eps}, min_samples={ms})",
                    "clusters": int(cluster_count),
                    "silhouette": sil,
                    "davies_bouldin": dbi,
                    "noise": noise_count,
                }
            )

    if best_model is None:
        raise RuntimeError("No clustering model could be trained.")

    labels = best_model.predict(X_scaled)
    cluster_sizes = {
        int(label): int((labels == label).sum()) for label in np.unique(labels)
    }

    feature_stats = _compute_feature_stats(X)

    return SegmentationArtifacts(
        imputer=imputer,
        scaler=scaler,
        model=best_model,
        model_name=f"KMeans (K={best_k})",
        k=int(best_k),
        metrics={"silhouette": float(best_sil), "davies_bouldin": float(best_dbi)},
        feature_names=X.columns.tolist(),
        feature_stats=feature_stats,
        cluster_sizes=cluster_sizes,
        benchmark=benchmark,
    )


def _compute_top_features(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict[str, float]]:
    model = pipeline.named_steps.get("model")
    if model is not None and hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        perm = permutation_importance(
            pipeline,
            X,
            y_encoded,
            n_repeats=5,
            random_state=42,
            scoring="f1_macro",
            n_jobs=-1,
        )
        importances = perm.importances_mean

    order = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "name": str(feature_names[int(idx)]),
            "importance": float(importances[int(idx)]),
        }
        for idx in order
    ]


def _train_recommendation(
    X: pd.DataFrame,
    y: pd.Series,
    classification: ClassificationArtifacts,
) -> RecommendationArtifacts:
    _, y_encoded, class_names = _encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    recommendation_models = _build_models()
    recommendation_models.pop(classification.model_name, None)
    if not recommendation_models:
        recommendation_models = {
            "Random Forest (Reco)": RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        }

    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for name, model in recommendation_models.items():
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
        raise RuntimeError("No recommendation model could be trained.")

    best_pipeline.fit(X, y_encoded)
    top_features = _compute_top_features(
        best_pipeline, X, y_encoded, classification.feature_names
    )

    return RecommendationArtifacts(
        pipeline=best_pipeline,
        feature_names=classification.feature_names,
        class_names=class_names,
        model_name=best_name,
        feature_stats=classification.feature_stats,
        top_features=top_features,
    )


class ModelService:
    def __init__(
        self,
        dataset_path: str | None = None,
        classification_path: str | None = None,
        segmentation_path: str | None = None,
        recommendation_path: str | None = None,
    ) -> None:
        self.dataset_path = dataset_path or DEFAULT_DATASET_PATH
        self.classification_path = classification_path or DEFAULT_CLASSIFIER_PATH
        self.segmentation_path = segmentation_path or DEFAULT_SEGMENTATION_PATH
        self.recommendation_path = recommendation_path or DEFAULT_RECOMMENDATION_PATH
        self._data_cache: Tuple[pd.DataFrame, pd.Series] | None = None

        self.classification = self._load_or_train_classification()
        self.segmentation = self._load_or_train_segmentation()
        self.recommendation = self._load_or_train_recommendation()

    def _get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._data_cache is None:
            csv_path = _find_dataset_file(self.dataset_path)
            X, y, _ = _prepare_dataframe(csv_path)
            self._data_cache = (X, y)
        return self._data_cache

    def _load_or_train_classification(self) -> ClassificationArtifacts:
        if os.path.isfile(self.classification_path):
            try:
                loaded = joblib.load(self.classification_path)
                if isinstance(loaded, ClassificationArtifacts):
                    return loaded
                if isinstance(loaded, ModelArtifacts):
                    return ClassificationArtifacts(
                        pipeline=loaded.pipeline,
                        feature_names=loaded.feature_names,
                        class_names=loaded.class_names,
                        model_name=loaded.model_name,
                        feature_stats=loaded.feature_stats,
                        f1_macro=0.0,
                    )
            except AttributeError:
                pass

        X, y = self._get_data()
        artifacts = _train_classification(X, y)
        joblib.dump(artifacts, self.classification_path)
        return artifacts

    def _load_or_train_segmentation(self) -> SegmentationArtifacts:
        if os.path.isfile(self.segmentation_path):
            return joblib.load(self.segmentation_path)

        X, _ = self._get_data()
        artifacts = _train_segmentation(X)
        joblib.dump(artifacts, self.segmentation_path)
        return artifacts

    def _load_or_train_recommendation(self) -> RecommendationArtifacts:
        if os.path.isfile(self.recommendation_path):
            return joblib.load(self.recommendation_path)

        X, y = self._get_data()
        artifacts = _train_recommendation(X, y, self.classification)
        joblib.dump(artifacts, self.recommendation_path)
        return artifacts

    def classification_metadata(self) -> Dict[str, object]:
        features = [
            {
                "name": name,
                "min": self.classification.feature_stats[name]["min"],
                "max": self.classification.feature_stats[name]["max"],
                "mean": self.classification.feature_stats[name]["mean"],
            }
            for name in self.classification.feature_names
        ]
        return {
            "features": features,
            "classes": self.classification.class_names,
            "model": self.classification.model_name,
            "f1_macro": self.classification.f1_macro,
        }

    def classification_predict(
        self, features: Dict[str, float], top_n: int = 3
    ) -> Dict[str, object]:
        missing = [
            name
            for name in self.classification.feature_names
            if name not in features
        ]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")

        ordered_values = [features[name] for name in self.classification.feature_names]
        df = pd.DataFrame([ordered_values], columns=self.classification.feature_names)
        probas = self.classification.pipeline.predict_proba(df)[0]
        sorted_indices = np.argsort(probas)[::-1][:top_n]

        top = [
            {
                "label": self.classification.class_names[int(idx)],
                "probability": float(probas[int(idx)]),
            }
            for idx in sorted_indices
        ]
        alert = float(probas[sorted_indices[0]]) < 0.5

        return {
            "top": top,
            "alert": alert,
            "model": self.classification.model_name,
        }

    def segmentation_metadata(self) -> Dict[str, object]:
        features = [
            {
                "name": name,
                "min": self.segmentation.feature_stats[name]["min"],
                "max": self.segmentation.feature_stats[name]["max"],
                "mean": self.segmentation.feature_stats[name]["mean"],
            }
            for name in self.segmentation.feature_names
        ]
        return {
            "features": features,
            "model": {
                "name": self.segmentation.model_name,
                "k": self.segmentation.k,
                "silhouette": self.segmentation.metrics["silhouette"],
                "davies_bouldin": self.segmentation.metrics["davies_bouldin"],
            },
            "cluster_sizes": self.segmentation.cluster_sizes,
            "benchmark": self.segmentation.benchmark,
        }

    def segmentation_predict(self, features: Dict[str, float]) -> Dict[str, object]:
        missing = [
            name
            for name in self.segmentation.feature_names
            if name not in features
        ]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")

        ordered_values = [features[name] for name in self.segmentation.feature_names]
        df = pd.DataFrame([ordered_values], columns=self.segmentation.feature_names)
        imputed = self.segmentation.imputer.transform(df)
        scaled = self.segmentation.scaler.transform(imputed)
        distances = self.segmentation.model.transform(scaled)[0]
        cluster = int(np.argmin(distances))
        cluster_size = int(self.segmentation.cluster_sizes.get(cluster, 0))

        return {
            "cluster": cluster,
            "model": self.segmentation.model_name,
            "k": self.segmentation.k,
            "distance": float(distances[cluster]),
            "cluster_size": cluster_size,
        }

    def recommendation_metadata(self) -> Dict[str, object]:
        features = [
            {
                "name": name,
                "min": self.recommendation.feature_stats[name]["min"],
                "max": self.recommendation.feature_stats[name]["max"],
                "mean": self.recommendation.feature_stats[name]["mean"],
            }
            for name in self.recommendation.feature_names
        ]
        return {
            "features": features,
            "classes": self.recommendation.class_names,
            "model": self.recommendation.model_name,
            "top_features": self.recommendation.top_features,
        }

    def recommendation_predict(
        self, features: Dict[str, float], top_n: int = 3
    ) -> Dict[str, object]:
        missing = [
            name
            for name in self.recommendation.feature_names
            if name not in features
        ]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")

        ordered_values = [features[name] for name in self.recommendation.feature_names]
        df = pd.DataFrame([ordered_values], columns=self.recommendation.feature_names)
        probas = self.recommendation.pipeline.predict_proba(df)[0]
        sorted_indices = np.argsort(probas)[::-1][:top_n]

        top = [
            {
                "label": self.recommendation.class_names[int(idx)],
                "probability": float(probas[int(idx)]),
            }
            for idx in sorted_indices
        ]
        alert = float(probas[sorted_indices[0]]) < 0.5

        return {
            "top": top,
            "alert": alert,
            "model": self.recommendation.model_name,
        }
