import geodatasets
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def plot_map(df_stores: pd.DataFrame):
    """_summary_

    Args:
        df_stores (pd.DataFrame): _description_

    Example usage:
        df_stores.columns = "store_address", "latitude", "longitude"
        usa = gpd.read_file(geodatasets.get_path("geoda us_sdoh")) = United States Map (documentation)
    """
    gdf = gpd.GeoDataFrame(
        df_stores,
        geometry=gpd.points_from_xy(df_stores["longitude"], df_stores["latitude"]),
    )

    # Lade die USA-Karte aus geodatasets
    usa = gpd.read_file(geodatasets.get_path("geoda us_sdoh"))

    # Plot die Karte mit den Store-Punkten
    fig, ax = plt.subplots(figsize=(20, 15))
    usa.plot(ax=ax)

    # Plot der Stores
    gdf.plot(ax=ax, color="red", markersize=50)

    plt.title("Store Locations in the United States")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
