import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_importance(
    model, feature_names, title="Feature Importance", palette="viridis", figsize=(10, 6)
):
    """
    Erstellt einen Barplot der Feature-Importances eines beliebigen Modells.

    Parameters:
        model: Ein trainiertes Modell mit dem Attribut 'feature_importances_' (z.B. RandomForest, XGBoost)
        feature_names: Liste der Namen der Features
        title: Titel des Plots (Default: "Feature Importance")
        palette: Farbpalette für den Plot (Default: "viridis")
        figsize: Tuple für die Plotgröße (Default: (10, 6))
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot erstellen
    plt.figure(figsize=figsize)

    sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices],
        hue=np.array(feature_names)[indices],
        palette=palette,
        legend=False,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    plt.tight_layout()
    plt.show()
