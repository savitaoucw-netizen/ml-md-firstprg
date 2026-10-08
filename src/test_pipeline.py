import os
import joblib
from src.data import generate_raw_tabular_data
from src.features import build_feature_pipeline, save_preprocessor


def test_feature_pipeline_transformation(tmp_path):
  X_train, X_test, _, _ = generate_raw_tabular_data(n_samples=200)
  num_cols = ["age", "income", "credit_score"]
  cat_cols = ["education", "city_tier"]

  pipeline = build_feature_pipeline(num_cols, cat_cols)

  # Fit and transform
  X_train_transformed = pipeline.fit_transform(X_train)
  X_test_transformed = pipeline.transform(X_test)

  # Assert no NaNs remain after imputation
  assert not any(
      hasattr(X_train_transformed, "isnull") and X_train_transformed.isnull().any()
  )

  # Verify number of columns expanded due to one-hot encoding
  # 3 numerical + 3 education one-hot + 3 city_tier one-hot = 9 columns
  assert X_train_transformed.shape[1] == 9
  assert X_test_transformed.shape[1] == 9

  # Verify serialization
  temp_save_path = os.path.join(tmp_path, "test_preprocessor.joblib")
  save_preprocessor(pipeline, temp_save_path)
  assert os.path.exists(temp_save_path)

  # Verify reloaded model inference
  reloaded_pipeline = joblib.load(temp_save_path)
  reloaded_transformed = reloaded_pipeline.transform(X_test)
  assert reloaded_transformed.shape == X_test_transformed.shape
