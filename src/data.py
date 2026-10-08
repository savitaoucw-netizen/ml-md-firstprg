
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_raw_tabular_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
):
  """Generates a synthetic raw tabular dataset with missing values and categorical features."""
  np.random.seed(random_state)

  raw_data = {
      "age": np.random.choice(
          [22, 28, 35, 42, np.nan, 55, 63], size=n_samples
      ),
      "income": np.random.choice(
          [35000, 52000, 78000, np.nan, 110000], size=n_samples
      ),
      "credit_score": np.random.normal(loc=680, scale=45, size=n_samples),
      "education": np.random.choice(
          ["High School", "Bachelor", "Master", np.nan], size=n_samples
      ),
      "city_tier": np.random.choice(
          ["Tier_1", "Tier_2", "Tier_3"], size=n_samples
      ),
      "target": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
  }

  df = pd.DataFrame(raw_data)

  X = df[["age", "income", "credit_score", "education", "city_tier"]]
  y = df["target"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state, stratify=y
  )

  return X_train, X_test, y_train, y_test
