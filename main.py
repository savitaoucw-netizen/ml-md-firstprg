import argparse
import yaml
from src.data import generate_raw_tabular_data
from src.features import build_feature_pipeline, save_preprocessor


def main():
  parser = argparse.ArgumentParser(
      description="Automated Feature Engineering Pipeline"
  )
  parser.add_argument(
      "--config",
      type=str,
      default="configs/config.yaml",
      help="Path to YAML config",
  )
  args = parser.parse_args()

  # Load YAML configs
  with open(args.config, "r") as f:
    config = yaml.safe_load(f)

  num_cols = config["features"]["numerical"]
  cat_cols = config["features"]["categorical"]
  preprocessor_path = config["artifacts"]["preprocessor_path"]

  print("Step 1: Ingesting Raw Tabular Data with Missing Values...")
  X_train, X_test, _, _ = generate_raw_tabular_data(
      n_samples=config["dataset"]["n_samples"],
      test_size=config["dataset"]["test_size"],
      random_state=config["dataset"]["random_state"],
  )

  print(f"Raw Input Matrix Shape: {X_train.shape}")
  print(f"Missing Values in Raw Training Data:\n{X_train.isnull().sum()}\n")

  print("Step 2: Fitting Automated Scikit-Learn Feature Pipeline...")
  pipeline = build_feature_pipeline(num_cols, cat_cols)
  X_train_transformed = pipeline.fit_transform(X_train)
  X_test_transformed = pipeline.transform(X_test)

  print(
      "✅ Missing values imputed, numericals scaled, categoricals one-hot"
      " encoded."
  )
  print(f"Transformed Feature Matrix Shape: {X_train_transformed.shape}\n")

  print("Step 3: Serializing Feature Pipeline Artifact...")
  save_preprocessor(pipeline, preprocessor_path)
  print(f"✅ Saved preprocessor to: {preprocessor_path}")


if __name__ == "__main__":
  main()
