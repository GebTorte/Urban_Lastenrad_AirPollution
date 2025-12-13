import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt

ox.settings.use_cache = True
ox.settings.log_console = True


"""
EXAMPLES

place_name = "Würzburg, Germany"
tags = {'building': True}


# Define the location and tags for loading data
place_name = "Würzburg, Germany"
tags = {'landuse': True}
"""


def load_osm_data(place: str, tags: dict):
    """
    Load and filter OSM data
    """
    # Use the latest function to fetch geographic data
    gdf = ox.features_from_place(place, tags)  # Use features_from_place in osmnx 2.0.0+
    return gdf



def load_and_filter_osm_data(place: str, tags: dict, filter: str | None = None):
    """
    Load and filter OSM data for a specific location.

    Parameters:
        place (str): The location name (e.g., "Würzburg, Germany").
        tags (dict): Tags to filter specific OSM features (e.g., {'landuse': True}).
        filter (str): "name", "landuse", etc.

    Returns:
        GeoDataFrame: The filtered GeoDataFrame.
    """
    # Load data using the updated OSMnx function
    gdf = ox.features_from_place(place, tags)  # Updated method for OSMnx 2.0+

    if filter:
    # Filter the required columns if they exist
        if filter in gdf.columns:
            gdf = gdf[['geometry', filter]].copy()
            # Remove rows where 'landuse' is null
            gdf = gdf[gdf[filter].notnull()]
        else:
            raise ValueError(f"'{filter}' column is missing in the data.")

    return gdf


def plot_data(gdf: gpd.GeoDataFrame):
    """
    Plot map data
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, alpha=0.6, edgecolor='k')
    ax.set_title('Result')
    plt.tight_layout()
    plt.show()

def save_gdf_to_file(gdf, file_path):
    """
    Save the GeoDataFrame to a specified file in GeoPackage format.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data to save.
        file_path (str): The path where the file will be saved.
    """
    if gdf.empty:
        print("No data to save.")
        return

    # Convert relative path to absolute path
    file_path = os.path.abspath(file_path)

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save the GeoDataFrame as a GeoPackage file
    gdf.to_file(file_path, driver='GPKG')
    print(f"GeoDataFrame successfully saved to {file_path}")

def plot_landuse_data(gdf, title: str = "Title"):
    """
    Plot the filtered GeoDataFrame with a legend.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data to plot.
    """
    if gdf.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        ax=ax,
        column='landuse',
        legend=True,
        legend_kwds={'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'}
    )
    ax.set_title(title) #'Land Use in Würzburg, Germany'
    plt.tight_layout(rect=(0, 0, 0.85, 1))  # Adjust layout for legend
    plt.show()

