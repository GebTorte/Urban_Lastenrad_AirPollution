# I'll create a comprehensive Python analysis script/notebook that you can use
# This will be presented as code to insert into your notebook system

"""
AIR POLLUTION & BUILDING HEIGHT CORRELATION ANALYSIS
=====================================================

LITERATURE-BASED APPROACH:
1. Urban Heat Island Effect: Building morphology affects air circulation (Santamouris et al., 2015)
2. Street Canyon Effect: Building height ratios affect pollutant dispersion (Oke, 1988)
3. Particle Concentration Patterns: Fine particles accumulate in street canyons with H/W > 1
   (where H = building height, W = street width) (Vardoulakis et al., 2003)
4. Vertical Mixing: Taller buildings create vortices that trap particles at street level
5. Methodology: Map particle measurements to nearest buildings, correlate with height

ANALYSIS OBJECTIVES:
- Correlate particle density (0.1L-1.0L diameter) with building heights
- Identify street canyon hotspots (high height/width ratio)
- Create spatial heatmaps of pollution distribution
- Statistical validation (R², Pearson correlation, Mann-Whitney tests)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import warnings

from svf_calculation import calculate_svf_easy_raycasting

warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================


def load_particle_measurements(
    data_dir="./data/standardized_withNA/", sensor_position="particleBack", epsg=4326
):
    """
    Load all particle measurement data and stack them.

    Args:
        data_dir: Directory containing standardized measurement files
        sensor_position: 'particleFront', 'particleBack', or 'particleBottom'

    Returns:
        GeoDataFrame with all measurements combined
    """
    df_list = []

    for gpkg_file in os.listdir(data_dir):
        if gpkg_file.endswith(".gpkg"):
            try:
                df = gpd.read_file(os.path.join(data_dir, gpkg_file))
                df = df.to_crs(epsg=epsg)

                # Filter for specific sensor position only
                particle_cols = [
                    col
                    for col in df.columns
                    if sensor_position in col and "Particles" in col
                ]
                keep_cols = ["geometry", "distance_from_start_km"] + particle_cols
                keep_cols = [c for c in keep_cols if c in df.columns]

                df_filtered = df[keep_cols]
                df_filtered["measurement_source"] = gpkg_file
                df_list.append(df_filtered)
                print(f"✓ Loaded {gpkg_file}")
            except Exception as e:
                print(f"✗ Error loading {gpkg_file}: {e}")

    if df_list:
        combined_gdf = pd.concat(df_list, ignore_index=True)
        combined_gdf = combined_gdf.reset_index(drop=True)
        return combined_gdf
    else:
        raise ValueError(f"No files found in {data_dir}")


def load_buildings(
    buildings_path="./data/osm/wue_buildings_and_landuse.gpkg",
    epsg: int = 4326,
    default_level_height: float = 3.5,
    default_levels: int = 3,
    default_roof_level_height: float = 2.5,
):
    """
    Load OSM building data with height information.

    Project to given EPSG.

    Returns:
        GeoDataFrame with osm buildings and calculated heights in "height_calc" column
    """
    buildings = gpd.read_file(buildings_path)
    buildings = buildings.to_crs(epsg=epsg)

    # TODO: determine default levels and height from data itself

    # Filter for building geometries only
    if "building" in buildings.columns:
        buildings = buildings[buildings["building"].notna()]

    # parse buildings height for string info
    buildings["height"] = (
        buildings["height"]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)", expand=False)
        .astype(float)
    )

    # Calculate building heights
    buildings["height_calc"] = buildings["height"].fillna(
        pd.to_numeric(buildings["building:levels"], errors="coerce").fillna(
            default_levels
        )
        * default_level_height  # 3.5m per level default
        + pd.to_numeric(buildings["roof:levels"], errors="coerce").fillna(0)
        * default_roof_level_height
    )

    # buildings["height_calc"] = pd.to_numeric(buildings["height_calc"], errors="coerce")
    buildings["height_calc"] = buildings["height_calc"]  # .clip(lower=3, upper=150)

    print(f"✓ Loaded {len(buildings)} buildings")
    return buildings


def load_GBA_buildings(buildings_path="./data/gba_wue.geojson", epsg: int = 4326):
    """
    Load GBA building data with height information.

    Project to given EPSG.

    Returns:
        GeoDataFrame with GBA buildings and calculated heights in "height_calc" column
    """
    buildings = gpd.read_file(buildings_path)
    buildings = buildings.to_crs(epsg=epsg)

    # rename height column
    buildings.rename(columns={"height": "height_calc"}, inplace=True)

    print(f"✓ Loaded {len(buildings)} GBA buildings")
    return buildings


# ============================================================================
# PART 2: SPATIAL MATCHING & FEATURE EXTRACTION
# ============================================================================

# Define raster resolution (in degrees - adjust as needed)
RASTER_RES = 1.0 / 111000  # ~1 meter


def create_2d_dem_raster_from_3d_buildings(gdf):
    """
    Docstring for create_2d_dem_raster_from_3d_buildings

    :param gdf: Description
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # Get bounds of the GeoDataFrame
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds

    # Create output grid dimensions
    width = int((maxx - minx) / RASTER_RES)
    height = int((maxy - miny) / RASTER_RES)

    # Create transform (mapping pixel coordinates to geographic coordinates)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Prepare geometries with height values for rasterization
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf["height_calc"])]

    # Rasterize: burn height_calc values into grid
    dem_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Background value for cells with no building
        dtype="float32",
        default_value=0,
    )

    return dem_raster


def extract_aggregated_particle_size_bins(gdf: gpd.GeoDataFrame):
    """
    Extract and aggregate particle measurements into size categories.

    Standard size bins from sensor: 0.3µm, 0.5µm, 1.0µm, 2.5µm, 5.0µm, 10.0µm
    Particles 0.3µm-1.0µm: Ultrafine (UFP) - health relevant
    Particles > 2.5µm: PM2.5 equivalent
    """
    particle_cols = [col for col in gdf.columns if "Particles" in col]

    # Map column names to size bins
    size_mapping = {}
    for col in particle_cols:
        for size in ["0.3um", "0.5um", "1.0um", "2.5um", "5.0um", "10.0um"]:
            if size in col:
                size_mapping[col] = size
                break

    result = pd.DataFrame()
    for size, cols in size_mapping.items():
        size_cols = [col for col in particle_cols if size in col]
        if size_cols:
            result[f"particles_{size}"] = gdf[size_cols].mean(axis=1, skipna=True)

    # Create composite indices
    result["ultrafine_ufp"] = gdf[
        [c for c in particle_cols if "0.3um" in c or "0.5um" in c or "1.0um" in c]
    ].mean(axis=1, skipna=True)
    result["pm25_equiv"] = gdf[
        [c for c in particle_cols if "2.5um" in c or "5.0um" in c or "10.0um" in c]
    ].mean(axis=1, skipna=True)

    return pd.concat([gdf[["geometry"]], result], axis=1)


def map_measurements_to_buildings(
    measurements_gdf, buildings_gdf, max_distance: float = 30.0, k: int = 20
):
    """
    For each measurement point, find nearest buildings and extract building features.

    Args:
        measurements_gdf: GeoDataFrame with particle measurements (points)
        buildings_gdf: GeoDataFrame with buildings (polygons)
        float max_distance: Maximum distance in meters to search for buildings (for street canyon effects etc)
        int k: max number of buildings to consider

    Returns:
        GeoDataFrame with measurements + nearest building features
    """
    # Use spatial indexing for efficiency
    building_tree = KDTree([[p.x, p.y] for p in buildings_gdf.geometry.centroid])
    measurement_coords = np.array([(p.x, p.y) for p in measurements_gdf.geometry])

    # Find k nearest buildings in buffer
    distances, indices = building_tree.query(
        measurement_coords, k=k, distance_upper_bound=max_distance
    )

    # Initialize result columns
    measurements_gdf["nearest_building_height"] = np.nan
    measurements_gdf["nearest_building_distance"] = np.nan
    measurements_gdf[f"mean_{k}nn_building_height"] = np.nan
    measurements_gdf[f"max_{k}nn_building_height"] = np.nan
    measurements_gdf["street_canyon_index"] = np.nan

    # TODO: add 30m nearest buildings (only first hit)
    # vectors in star form outgoing from point
    # take set of resulting buildings

    for i, (dist, idx_list) in enumerate(zip(distances, indices)):
        valid_buildings = buildings_gdf.iloc[
            idx_list[dist <= max_distance]
        ]  # convert m to degrees

        if len(valid_buildings) > 0:
            heights = valid_buildings["height_calc"].values
            measurements_gdf.loc[i, "nearest_building_height"] = heights[0]
            measurements_gdf.loc[i, "nearest_building_distance"] = dist[0]
            measurements_gdf.loc[i, f"mean_{k}nn_building_height"] = np.nanmean(heights)
            measurements_gdf.loc[i, f"max_{k}nn_building_height"] = np.nanmax(heights)

    return measurements_gdf


def calculate_urban_morphology_indices(
    measurements_gdf, buildings_gdf, svf_raster=None, buffer_radius=30
):
    """
    Calculate urban morphology indices (SVF, aspect ratio, building density).

    Based on: Oke (1988), Stewart & Oke (2012) Local Climate Zone methodology
    """
    measurements_gdf["sky_view_factor"] = np.nan
    measurements_gdf["building_surface_fraction"] = np.nan
    measurements_gdf["building_density"] = np.nan

    # TODO:
    measurements_gdf["mean_building_height_in_buffer"] = np.nan

    for idx, point in measurements_gdf.iterrows():
        print(
            f"Progess on calculating urban morpohogy indices {idx}/{len(measurements_gdf)}",
            end="\r",
        )
        # Create circular buffer around measurement point
        buffer = point.geometry.buffer(
            buffer_radius
        )  # buffer_radius / 111000)  # convert m to degrees

        # Buildings within buffer
        buildings_in_buffer = buildings_gdf[buildings_gdf.geometry.intersects(buffer)]

        if len(buildings_in_buffer) > 0:
            # Building density: count per 100m radius

            # Building surface fraction: total building area / buffer area
            building_area = buildings_in_buffer.geometry.area.sum()
            # buffer_area = np.pi * (buffer_radius / 111000) ** 2
            buffer_area = np.pi * buffer_radius**2

            measurements_gdf.loc[idx, "building_surface_fraction"] = (
                building_area / buffer_area
            )

            # Sky view factor approximation (inverse of building surface fraction)
            measurements_gdf.loc[idx, "sky_view_factor"] = 1 - (
                building_area / buffer_area
            )

            measurements_gdf.loc[idx, "mean_building_height_in_buffer"] = np.nanmean(
                buildings_in_buffer["height_calc"].values
            )

            measurements_gdf.loc[idx, "median_building_height_in_buffer"] = (
                np.nanmedian(buildings_in_buffer["height_calc"].values)
            )

            # sky view factor dem based
            # TODO: get neareast point that exists in raster
            # if svf_raster:
            # measurements_gdf.loc[idx, "sky_view_factor_dem"] = svf_raster[point.x, point.y] # TODO: get points svf from svf arr

            # sky view factor raytracing based
            measurements_gdf.loc[idx, "svf_ray"] = calculate_svf_easy_raycasting(
                point.geometry,
                buildings_gdf,
                max_distance=RAY_LENGTH,
                azimuth_divisions=60,
            )

    return measurements_gdf


# ============================================================================
# PART 3: STATISTICAL ANALYSIS
# ============================================================================


def perform_correlation_analysis(measurements_gdf, output_dir="./results/"):
    """
    Perform comprehensive statistical analysis of pollution vs building height.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Get particle columns
    particle_cols = [
        col
        for col in measurements_gdf.columns
        if "particles_" in col or "ufp" in col or "pm25" in col
    ]
    building_cols = [
        "nearest_building_height",
        "mean_20nn_building_height",
        "max_20nn_building_height",
        "building_surface_fraction",
        "mean_building_height_in_buffer",
        "median_building_height_in_buffer",
        "svf_ray",
    ]

    # Remove rows with NaN in key columns
    analysis_df = measurements_gdf[particle_cols + building_cols].dropna()

    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: AIR POLLUTION vs BUILDING MORPHOLOGY")
    print("=" * 70)

    correlation_matrix = analysis_df.corr()

    # Extract key correlations
    for pollutant in particle_cols:
        for morph in building_cols:
            if (
                pollutant in correlation_matrix.index
                and morph in correlation_matrix.columns
            ):
                corr_value = correlation_matrix.loc[pollutant, morph]

                # Pearson correlation test
                valid_data = analysis_df[[pollutant, morph]].dropna()
                r, p_value = stats.pearsonr(valid_data[pollutant], valid_data[morph])

                results[f"{pollutant}_vs_{morph}"] = {
                    "correlation": r,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

                significance = (
                    "***"
                    if p_value < 0.001
                    else "**"
                    if p_value < 0.01
                    else "*"
                    if p_value < 0.05
                    else "ns"
                )
                print(
                    f"{pollutant:20s} vs {morph:30s}: r={r:7.4f}, p={p_value:.4e} {significance}"
                )

    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation"},
    )
    plt.title("Correlation Matrix: Air Pollution vs Urban Morphology")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return results, analysis_df


# ============================================================================
# PART 4: SPATIAL VISUALIZATION
# ============================================================================


def create_pollution_heatmap(
    measurements_gdf,
    buildings_gdf,
    pollutant_col="ultrafine_ufp",
    grid_resolution=100,
    output_path="pollution_heatmap.html",
):
    """
    Create interactive heatmap of pollution with building overlay.

    Uses Folium for interactive map.
    """
    import folium
    from folium import plugins

    # go to lat lon for foloium
    measurements_gdf = measurements_gdf.copy().to_crs(epsg=4326)

    # Prepare data for heatmap
    valid_data = measurements_gdf[measurements_gdf[pollutant_col].notna()]

    if len(valid_data) < 3:
        print(f"Not enough valid data points for heatmap")
        return None

    # Create base map
    center_lat = valid_data.geometry.y.mean()
    center_lon = valid_data.geometry.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
    )

    # Normalize pollutant values for color mapping
    vmin = valid_data[pollutant_col].quantile(0.05)
    vmax = valid_data[pollutant_col].quantile(0.95)

    # Add heatmap layer
    heat_data = [
        [row.geometry.y, row.geometry.x, row[pollutant_col]]
        for idx, row in valid_data.iterrows()
    ]

    # TODO make heatmap interactive (choose the column for colorization, multiple maps at the same time (toggle), etc.)

    plugins.HeatMap(heat_data, radius=15, blur=25, max_zoom=1).add_to(m)

    # Add measurement points as markers
    for idx, row in valid_data.iterrows():
        color_intensity = (row[pollutant_col] - vmin) / (vmax - vmin)
        color_intensity = np.clip(color_intensity, 0, 1)

        # Color from blue (low) to red (high)
        color = f"hsl({int((1 - color_intensity) * 240)}, 100%, 50%)"

        html = folium.Html(f"""<html><p>Particles ufp: {row[pollutant_col]:.1f}<br />
Particles pm25 equivalent : {row["pm25_equiv"]:.1f}<br />
sky_view factor: {row["svf_ray"]:.2f}<br />
Mean Building height 20nn {row["mean_20nn_building_height"]:1f}<br />
mean_building_height_in_buffer: {row["mean_building_height_in_buffer"]:1f}
</p></html>""")

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            popup=folium.Popup(html=html),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1,
        ).add_to(m)

    # Add buildings as semi-transparent polygons
    for idx, building in buildings_gdf.iterrows():
        height = building["height_calc"]
        opacity = min(height / 50, 0.5)  # More opaque for taller buildings

        if hasattr(building.geometry, "exterior"):
            coords = building.geometry.exterior.coords
        elif hasattr(building.geometry, "geoms"):
            # MultiPolygon - use first polygon's exterior
            coords = building.geometry.geoms[0].exterior.coords
        else:
            coords = building.geometry.coords
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in coords],
            color="gray",
            fill=True,
            fillColor="gray",
            fillOpacity=opacity,
            weight=1,
            popup=f"Height: {height:.1f}m",
        ).add_to(m)

    m.save(output_path)
    print(f"✓ Heatmap saved to {output_path}")
    return m


def create_building_height_pollution_plot(
    measurements_gdf, output_path="building_height_vs_pollution.png"
):
    """
    Create scatter plots and regression analyses for key relationships.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    particle_cols = [col for col in measurements_gdf.columns if "particles_" in col][:4]

    for idx, col in enumerate(particle_cols):
        ax = axes[idx // 2, idx % 2]

        # Remove NaN values
        valid = measurements_gdf[["nearest_building_height", col]].dropna()

        if len(valid) > 10:
            # Scatter plot
            ax.scatter(valid["nearest_building_height"], valid[col], alpha=0.5, s=30)

            # Regression line
            z = np.polyfit(valid["nearest_building_height"], valid[col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(
                valid["nearest_building_height"].min(),
                valid["nearest_building_height"].max(),
                100,
            )
            ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Linear fit")

            # Calculate R²
            r_squared = (
                np.corrcoef(valid["nearest_building_height"], valid[col])[0, 1] ** 2
            )

            ax.set_xlabel("Building Height (m)")
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_title(
                f"{col.replace('_', ' ').title()} vs Building Height (R²={r_squared:.3f})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Scatter plots saved to {output_path}")


def create_svf_vs_pollution_plot(
    measurements_gdf,
    output_path="svf_vs_pollution.png",
    columns_matching=["particles_", "ultrafine_ufp", "pm25_equiv"],
):
    """
    Create scatter plots and regression analyses for key relationships.
    """
    particle_cols = [
        col
        for col in measurements_gdf.columns
        if any(c in col for c in columns_matching)
    ]

    fig, axes = plt.subplots(len(particle_cols), 1, figsize=(6, 7 * len(particle_cols)))

    variable_col = "svf_ray"

    for idx, col in enumerate(particle_cols):
        ax = axes[idx]

        # Remove NaN values
        valid = measurements_gdf[[variable_col, col]].dropna()

        if len(valid) > 10:
            # Scatter plot
            ax.scatter(valid[variable_col], valid[col], alpha=0.5, s=30)

            # Regression line
            z = np.polyfit(valid[variable_col], valid[col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(
                valid[variable_col].min(),
                valid[variable_col].max(),
                100,
            )
            ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Linear fit")

            # Calculate R²
            # effect strength
            r_squared = np.corrcoef(valid[variable_col], valid[col])[0, 1] ** 2

            # Calculate Cohen's d (effect size)
            mean_diff = valid[variable_col].mean() - valid[col].mean()
            pooled_std = np.sqrt(
                (valid[variable_col].std() ** 2 + valid[col].std() ** 2) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

            # robust distribution-free effect size d_reg
            # Calculate robust distribution-free effect size (rank-biserial correlation)
            ranked_col = stats.rankdata(valid[col])
            n = len(valid)
            r_rb = 1 - (2 * ranked_col.sum()) / (n * (n + 1))
            d_reg = r_rb * 2  # Convert rank-biserial to Cohen's d equivalent

            # Calculate rank-biserial correlation (distribution-free effect size)
            n = len(valid)
            rank_data = stats.rankdata(valid[variable_col])
            r_delta = 1 - (2 * rank_data.sum()) / (n * (n + 1))

            # Calculate Spearman correlation (non-parametric alternative to Pearson)
            spearman_r, spearman_p = stats.spearmanr(valid[variable_col], valid[col])

            ax.set_xlabel(variable_col)
            ax.set_ylabel(col.replace("_", " ").title())
            ax.set_title(
                f"{col.replace('_', ' ').title()} vs {variable_col} (R²={r_squared:.3f}, cohens d = {cohens_d}, spearman r,p {spearman_r}, {spearman_p}, d_reg: {d_reg})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Scatter plots saved to {output_path}")


def create_density_heatmap_gridded(
    measurements_gdf,
    particle_col="ultrafine_ufp",
    grid_size=100,
    output_path="density_heatmap_grid.png",
):
    """
    Create gridded heatmap using spatial interpolation.
    """
    valid_data = (
        measurements_gdf[measurements_gdf[particle_col].notna()]
        .copy()
        .to_crs(epsg=4326)
    )

    # Create grid
    lon_min, lon_max = valid_data.geometry.x.min(), valid_data.geometry.x.max()
    lat_min, lat_max = valid_data.geometry.y.min(), valid_data.geometry.y.max()

    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    # Interpolate particle data
    points = np.array([valid_data.geometry.x, valid_data.geometry.y]).T
    values = valid_data[particle_col].values

    Z = griddata(points, values, (LON, LAT), method="cubic")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.contourf(LON, LAT, Z, levels=20, cmap="RdYlBu_r")
    ax.contour(LON, LAT, Z, levels=10, colors="black", alpha=0.2, linewidths=0.5)

    # Add measurement points
    scatter = ax.scatter(
        valid_data.geometry.x,
        valid_data.geometry.y,
        c=valid_data[particle_col],
        cmap="RdYlBu_r",
        s=50,
        edgecolors="black",
        linewidth=0.5,
        alpha=0.7,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Interpolation: {particle_col.replace('_', ' ').title()}")
    cbar = plt.colorbar(im, ax=ax, label="Particle Concentration (#/0.1L)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Gridded heatmap saved to {output_path}")


# ============================================================================
# PART 5: STREET CANYON ANALYSIS
# ============================================================================


def analyze_street_canyons(
    measurements_gdf, buildings_gdf, output_dir="./results/", quantile=0.75
):
    """
    Identify and analyze potential street canyon effects.

    High pollution areas with high building height ratios indicate
    potential street canyon effects trapping particles.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Classify measurements by morphology
    # TODO: other metric than nearest building height
    q75 = measurements_gdf["nearest_building_height"].quantile(quantile)
    q75_median = measurements_gdf["median_building_height_in_buffer"].quantile(quantile)
    q75_mean = measurements_gdf["mean_building_height_in_buffer"].quantile(quantile)
    q75_surface_fraction = measurements_gdf["building_surface_fraction"].quantile(
        quantile
    )

    q25 = measurements_gdf["ultrafine_ufp"].quantile(0.25)
    q75_ufp = measurements_gdf["ultrafine_ufp"].quantile(0.75)

    # street_canyon_mask = (measurements_gdf["nearest_building_height"] > q75) & (
    #    measurements_gdf["ultrafine_ufp"] > q75_ufp
    # )
    street_canyon_mask = (
        (measurements_gdf["nearest_building_height"] > q75)
        | (measurements_gdf["median_building_height_in_buffer"] > q75_median)
        | (measurements_gdf["mean_building_height_in_buffer"] > q75_mean)
        | (measurements_gdf["building_surface_fraction"] > q75_surface_fraction)
    )

    canyon_measurements = measurements_gdf[street_canyon_mask]

    print("\n" + "=" * 70)
    print("STREET CANYON HOTSPOT ANALYSIS")
    print("=" * 70)
    print(
        f"High-risk locations (High height + High pollution): {len(canyon_measurements)}"
    )
    print(
        f"Mean building height in canyons: {canyon_measurements['nearest_building_height'].mean():.1f}m"
    )
    print(
        f"Mean ultrafine particle concentration: {canyon_measurements['ultrafine_ufp'].mean():.1f} #/0.1L"
    )

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        measurements_gdf["nearest_building_height"],
        measurements_gdf["ultrafine_ufp"],
        c="blue",
        alpha=0.3,
        s=30,
        label="All measurements",
    )
    ax.scatter(
        canyon_measurements["nearest_building_height"],
        canyon_measurements["ultrafine_ufp"],
        c="red",
        s=50,
        label="Street canyon hotspots",
        edgecolors="darkred",
        linewidth=1,
    )

    ax.axvline(
        q75, color="gray", linestyle="--", label=f"75th percentile height ({q75:.1f}m)"
    )
    ax.axhline(
        q75_ufp,
        color="orange",
        linestyle="--",
        label=f"75th percentile UFP ({q75_ufp:.1f})",
    )

    ax.set_xlabel("Nearest Building Height (m)")
    ax.set_ylabel("Ultrafine Particle Concentration (#/0.1L)")
    ax.set_title("Street Canyon Effect: Identifying High-Risk Pollution Zones")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "street_canyon_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return canyon_measurements


# ============================================================================
# PART 6: COMPREHENSIVE REPORTING
# ============================================================================


def generate_analysis_report(
    measurements_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    output_dir="./results/",
):
    """
    Run complete analysis pipeline and generate report.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE AIR POLLUTION & URBAN MORPHOLOGY ANALYSIS")
    print("Wuerzburg Mobile Sensor Campaign - June 2024")
    print("=" * 70 + "\n")

    # Step 1: Data aggregation
    print("[1/6] Aggregating particle measurements...")
    measurements_gdf = extract_aggregated_particle_size_bins(measurements_gdf)

    # Step 2: Spatial matching
    print("[2/6] Matching measurements to building data...")
    measurements_gdf = map_measurements_to_buildings(measurements_gdf, buildings_gdf)

    # print("[3.0/6] Preparing SVF for raster"
    # import rvt
    # import rvt.vis
    # prepare rasterized "DEM"
    # dem = create_2d_dem_raster_from_3d_buildings(buildings_gdf)
    # out_dict= rvt.vis.sky_view_factor(
    #            dem,
    #            resolution=RASTER_RES,
    #            compute_svf=True,
    #            svf_r_max=10,
    #            svf_n_dir = 16,
    #        )
    # svf_arr = out_dict["svf"]

    # Step 3: Urban morphology
    print("[3/6] Calculating urban morphology indices...")
    measurements_gdf = calculate_urban_morphology_indices(
        measurements_gdf, buildings_gdf, buffer_radius=BUFFER_RADIUS
    )
    print(f"Saving temp measurements gdf to {output_dir}")
    measurements_gdf.to_file(
        os.path.join(output_dir, "measurements_with_building_features_temp.gpkg")
    )

    # exit("Exiting early")

    # Step 4: Statistical analysis
    print("[4/6] Performing correlation analysis...")
    results, analysis_df = perform_correlation_analysis(measurements_gdf, output_dir)

    # Step 5: Visualizations
    print("[5/6] Creating visualizations...")
    create_building_height_pollution_plot(
        measurements_gdf, os.path.join(output_dir, "building_height_vs_pollution.png")
    )

    create_svf_vs_pollution_plot(
        measurements_gdf, os.path.join(output_dir, "svf_ray_vs_pollution.png")
    )

    create_density_heatmap_gridded(
        measurements_gdf,
        "ultrafine_ufp",
        output_path=os.path.join(output_dir, "ufp_heatmap.png"),
    )
    create_density_heatmap_gridded(
        measurements_gdf,
        "pm25_equiv",
        output_path=os.path.join(output_dir, "pm25_heatmap.png"),
    )
    create_pollution_heatmap(
        measurements_gdf,
        buildings_gdf,
        output_path=os.path.join(output_dir, "interactive_pollution_map.html"),
    )

    # Step 6: Street canyon analysis
    print("[6/6] Analyzing street canyon effects...")
    canyon_data = analyze_street_canyons(measurements_gdf, buildings_gdf, output_dir)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total measurement points analyzed: {len(measurements_gdf)}")
    print(
        f"Mean nearest building height: {measurements_gdf['nearest_building_height'].mean():.1f}m"
    )
    print(
        f"Mean ultrafine particle concentration: {measurements_gdf['ultrafine_ufp'].mean():.1f} #/0.1L"
    )
    print(f"Mean PM2.5 equivalent: {measurements_gdf['pm25_equiv'].mean():.1f} #/0.1L")
    print(
        f"Building density (buildings/ha): {measurements_gdf['building_density'].mean():.2f}"
    )

    # Save processed data
    measurements_gdf.to_file(
        os.path.join(output_dir, "measurements_with_building_features.gpkg")
    )
    print(f"\n✓ Analysis complete. Results saved to {output_dir}/")

    return measurements_gdf, buildings_gdf


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # svf physics based ray length
    RAY_LENGTH = 100

    # buffer radius defines area for spatial metrics except svf ray
    BUFFER_RADIUS = 50

    utm_epsg = 32632  # important to convert to non 43- projection

    sensor_position = "particleBottom"

    # Load data
    measurements = load_particle_measurements(
        data_dir="./data/standardized_withNA",
        sensor_position=sensor_position,
        epsg=utm_epsg,
    )

    # osm buildings
    # buildings = load_buildings(epsg=utm_epsg)

    # loads gba buildings
    buildings = load_GBA_buildings(epsg=utm_epsg)
    # print(buildings)
    # print(buildings.columns)

    # Run analysis
    analyzed_data, buildings_data = generate_analysis_report(
        measurements, buildings, output_dir=f"./results/{sensor_position}/"
    )
