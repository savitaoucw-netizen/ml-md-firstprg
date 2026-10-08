import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_pipeline(
    num_features: list, cat_features: list
) -> ColumnTransformer:
  """Builds an automated Scikit-Learn ColumnTransformer pipeline.

  - Numerical Pipeline: Median Imputation -> Standard Scaling
  - Categorical Pipeline: Most Frequent Imputation -> One-Hot Encoding
  """
  # 1. Pipeline for Numerical Features
  num_pipeline = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
      ]
  )

  # 2. Pipeline for Categorical Features
  cat_pipeline = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          (
              "encoder",
              OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          ),
      ]
  )

  # 3. Combine both into a ColumnTransformer
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", num_pipeline, num_features),
          ("cat", cat_pipeline, cat_features),
      ],
      remainder="drop",
  )

  return preprocessor


def save_preprocessor(preprocessor, path: str = "feature_pipeline.joblib"):
  """Serializes fitted preprocessor pipeline to disk."""
  joblib.dump(preprocessor, path)


def load_preprocessor(path: str = "feature_pipeline.joblib"):
  """Loads serialized preprocessor pipeline from disk."""
  return joblib.load(path)
