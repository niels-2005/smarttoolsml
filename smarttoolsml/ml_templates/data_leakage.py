import pandas as pd


def check_data_leakage(X_train, X_test):
    overlap = pd.merge(X_train, X_test, how="inner")

    if not overlap.empty:
        print(
            f"Data leakage detected! {len(overlap)} overlapping rows found. Removing from X_train..."
        )
        print(f"X_train Shape {X_train.shape[0]} before")

        combined = pd.concat([X_train, X_test]).drop_duplicates(keep=False)

        X_train = combined.loc[combined.index.isin(X_train.index)]

        print(f"X_train Shape {X_train.shape[0]} after")
    else:
        print("No overlapping rows detected.")
