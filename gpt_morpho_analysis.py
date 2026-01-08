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
    data_dir="./data/standardized_withNA/", sensor_position="particleFront"
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
                df = df.to_crs(epsg=4326)

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


def load_buildings(buildings_path="./data/osm/wue_buildings_and_landuse.gpkg"):
    """
    Load building data with height information.

    Returns:
        GeoDataFrame with buildings, including calculated heights
    """
    buildings = gpd.read_file(buildings_path)
    buildings = buildings.to_crs(epsg=4326)

    # Filter for building geometries only
    if "building" in buildings.columns:
        buildings = buildings[buildings["building"].notna()]

    # Calculate building heights
    buildings["height_calc"] = buildings["height"].fillna(
        pd.to_numeric(buildings["building:levels"], errors="coerce").fillna(3.0)
        * 3.5  # 3.5m per level default
    )
    buildings["height_calc"] = pd.to_numeric(buildings["height_calc"], errors="coerce")
    buildings["height_calc"] = buildings["height_calc"].clip(lower=3, upper=150)

    print(f"✓ Loaded {len(buildings)} buildings")
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
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf['height_calc'])]

    # Rasterize: burn height_calc values into grid
    dem_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Background value for cells with no building
        dtype='float32',
        default_value=0
    )

    return dem_raster



def extract_aggregated_particle_size_bins(gdf):
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
    measurements_gdf, buildings_gdf, max_distance: float = 30.0, k: int = 10
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

    # Find 5 nearest buildings for each measurement
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
            idx_list[dist <= max_distance / 111000]
        ]  # convert m to degrees

        if len(valid_buildings) > 0:
            heights = valid_buildings["height_calc"].values
            measurements_gdf.loc[i, "nearest_building_height"] = heights[0]
            measurements_gdf.loc[i, "nearest_building_distance"] = (
                dist[0] * 111000
            )  # convert to meters
            measurements_gdf.loc[i, "mean_5nn_building_height"] = np.nanmean(heights)
            measurements_gdf.loc[i, "max_5nn_building_height"] = np.nanmax(heights)

    return measurements_gdf


def calculate_urban_morphology_indices(
    measurements_gdf, buildings_gdf, svf_raster=None, buffer_radius=100
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
        print(f"Progess on calculating urban morpohogy indices {idx}/{len(measurements_gdf)}", end="\r")
        # Create circular buffer around measurement point
        buffer = point.geometry.buffer(buffer_radius) #buffer_radius / 111000)  # convert m to degrees

        # Buildings within buffer
        buildings_in_buffer = buildings_gdf[buildings_gdf.geometry.intersects(buffer)]

        if len(buildings_in_buffer) > 0:
            # Building density: count per 100m radius
            measurements_gdf.loc[idx, "building_density"] = len(buildings_in_buffer) / (
                (buffer_radius**2) / 10000
            )

            # Building surface fraction: total building area / buffer area
            building_area = buildings_in_buffer.geometry.area.sum()
            buffer_area = np.pi * (buffer_radius / 111000) ** 2
            measurements_gdf.loc[idx, "building_surface_fraction"] = (
                building_area / buffer_area
            )

            # Sky view factor approximation (inverse of building surface fraction)
            measurements_gdf.loc[idx, "sky_view_factor"] = 1 - (
                building_area / buffer_area
            )

            # sky view factor dem based
            # TODO: get neareast point that exists in raster
            #if svf_raster:
                #measurements_gdf.loc[idx, "sky_view_factor_dem"] = svf_raster[point.x, point.y] # TODO: get points svf from svf arr

            # sky view factor raytracing based
            measurements_gdf.loc[idx, "svf_ray"] = calculate_svf_easy_raycasting(
                point.geometry, 
                buildings_gdf, 
                max_distance=30, 
                azimuth_divisions=90
            )

    return measurements_gdf

# ============================================================================
# PART 2B: PHYSICS-BASED SKY VIEW FACTOR CALCULATION
# ============================================================================
# Based on solid angle geometry and radiative view factor theory
# References: Oke (1988), Watson & Johnson (1987), Bourbia & Bosseur (2007)

def calculate_solid_angle_of_building(observation_point_coords, building_polygon, 
                                     building_height):
    """
    Calculate the solid angle subtended by a building as seen from an observation point.
    
    Physics Background:
    - Solid angle Ω measures how much of the hemisphere (2π steradians) is blocked
    - Sky View Factor (SVF) = (2π - Ω_total) / (2π)
    - Ω = ∫∫ cos(θ) dA / r² where θ is zenith angle, r is distance, dA is area element
    
    Args:
        observation_point_coords: tuple (lon, lat) of observation point
        building_polygon: Shapely polygon of building footprint
        building_height: float, height of building in meters
    
    Returns:
        float: solid angle in steradians (0 to 2π)
    """
    from shapely.geometry import Point, Polygon
    
    obs_lon, obs_lat = observation_point_coords
    
    # Convert building footprint to UTM coordinates for accurate distance calculation
    # This avoids lat/lon projection distortions
    def lonlat_to_utm_approx(lon, lat, ref_lon, ref_lat):
        """Approximate local UTM projection using reference point"""
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * np.cos(np.radians(lat))
        
        x = (lon - ref_lon) * meters_per_degree_lon
        y = (lat - ref_lat) * meters_per_degree_lat
        return x, y
    
    # Get building footprint vertices
    if hasattr(building_polygon, 'exterior'):
        building_coords = list(building_polygon.exterior.coords)[:-1]  # Remove duplicate closing point
    else:
        return 0.0  # Invalid polygon
    
    # Convert to local coordinates (meters, with observation point as origin)
    building_2d = np.array([lonlat_to_utm_approx(lon, lat, obs_lon, obs_lat) 
                            for lon, lat in building_coords])
    
    obs_point_2d = np.array([0.0, 0.0])
    
    # Calculate horizontal distances from observation point to each vertex
    distances = np.linalg.norm(building_2d - obs_point_2d, axis=1)
    
    # Avoid zero distances
    distances = np.where(distances < 0.1, 0.1, distances)
    
    # Calculate elevation angles to roofline and groundline
    # Elevation angle α: tan(α) = height / horizontal_distance
    elevation_angles_roof = np.arctan2(building_height, distances)  # top of building
    elevation_angles_ground = np.arctan(0.0 / distances)  # ground level ≈ 0
    
    # SOLID ANGLE CALCULATION using spherical projection
    # For a vertical rectangle subtending angles: azimuth(φ), elevation(α)
    # Ω ≈ Δφ × Δsin(α) for small angles
    
    # Calculate azimuths to each vertex from observation point
    azimuths = np.arctan2(building_2d[:, 1], building_2d[:, 0])  # atan2(y, x)
    
    # Compute azimuth differences (handle wrapping)
    azimuth_diffs = np.diff(np.concatenate([azimuths, [azimuths[0]]]))
    azimuth_diffs = np.abs(np.where(azimuth_diffs > np.pi, azimuth_diffs - 2*np.pi, azimuth_diffs))
    
    # Contribution to solid angle from each edge of building polygon
    solid_angle = 0.0
    
    for i in range(len(building_2d)):
        # Current and next vertex
        alpha1 = elevation_angles_roof[i]
        alpha2 = elevation_angles_roof[(i + 1) % len(building_2d)]
        phi = azimuth_diffs[i]
        
        # Solid angle element: Ω = Δφ × (sin(α2) - sin(α1))
        # This integrates cos(α) dα dφ over the rectangular patch
        delta_sin_alpha = np.sin(alpha2) - np.sin(alpha1)
        
        # Contribution capped at 0 to avoid negative values from geometry
        contribution = max(0, phi * delta_sin_alpha)
        solid_angle += contribution
    
    return solid_angle


def calculate_svf_from_buildings_geometry(
    observation_point, 
    buildings_gdf, 
    search_radius_m=100,
    angular_resolution_deg=5,
    num_azimuth_sectors=72,  # 360/5 = 72 sectors
):
    """
    Calculate Sky View Factor (SVF) using raycasting + building geometry.
    
    Method: Discretize the hemisphere into azimuth sectors and elevation angles.
    For each ray direction, find the highest building blocking that ray.
    SVF = (number of unblocked rays) / (total rays)
    
    Physics basis:
    - Sky dome = 2π steradians (full hemisphere)
    - SVF ∈ [0, 1]: 0 = completely enclosed, 1 = completely open sky
    - Values typical in cities: 0.3-0.8 (vs 1.0 in open fields)
    
    Args:
        observation_point: Shapely Point at observer location
        buildings_gdf: GeoDataFrame with building polygons and heights
        search_radius_m: Only consider buildings within this distance
        angular_resolution_deg: Resolution of angular discretization (degrees)
        num_azimuth_sectors: Number of azimuth directions (360/num = degree spacing)
    
    Returns:
        float: Sky View Factor (0 to 1)
        dict: Debugging info (max elevation per azimuth, etc.)
    """
    
    obs_lon, obs_lat = observation_point.x, observation_point.y
    
    def lonlat_to_meters(lon, lat, ref_lon, ref_lat):
        """Convert lat/lon differences to approximate meters"""
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * np.cos(np.radians((lat + ref_lat) / 2))
        x = (lon - ref_lon) * meters_per_degree_lon
        y = (lat - ref_lat) * meters_per_degree_lat
        return x, y
    
    # Filter buildings within search radius
    buildings_nearby = buildings_gdf.copy()
    buildings_nearby['dist_to_point'] = buildings_nearby.geometry.centroid.distance(observation_point)
    buildings_nearby = buildings_nearby[buildings_nearby['dist_to_point'] * 111000 <= search_radius_m]
    
    if len(buildings_nearby) == 0:
        return 1.0, {'num_buildings': 0, 'max_elevation_angles': []}
    
    # Create azimuth directions (0 to 360 degrees)
    azimuth_directions = np.linspace(0, 360, num_azimuth_sectors, endpoint=False)
    
    # For each azimuth direction, find maximum elevation angle to horizon
    max_elevation_angles = []
    
    for azimuth_deg in azimuth_directions:
        azimuth_rad = np.radians(azimuth_deg)
        max_elevation = 0.0
        
        # Ray direction in 2D (meters)
        ray_dir = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad)])
        
        # Check all nearby buildings
        for idx, building in buildings_nearby.iterrows():
            height = building['height_calc']
            
            # Get building footprint as polygon
            if hasattr(building.geometry, 'exterior'):
                building_coords = list(building.geometry.exterior.coords)[:-1]
            elif hasattr(building.geometry, 'geoms'):
                # MultiPolygon - use first polygon
                building_coords = list(building.geometry.geoms[0].exterior.coords)[:-1]
            else:
                continue
            
            # Convert to local coordinates (meters)
            building_2d = np.array([lonlat_to_meters(lon, lat, obs_lon, obs_lat)
                                   for lon, lat in building_coords])
            
            # For each edge of building, calculate intersection with ray
            for i in range(len(building_2d)):
                v1 = building_2d[i]
                v2 = building_2d[(i + 1) % len(building_2d)]
                
                # Find closest point on building edge to ray
                # Project edge onto ray direction
                edge_vec = v2 - v1
                edge_len = np.linalg.norm(edge_vec)
                
                if edge_len < 0.01:  # Skip degenerate edges
                    continue
                
                edge_dir = edge_vec / edge_len
                
                # Closest point on infinite line containing edge
                v1_rel = v1  # relative to observation point at origin
                t = np.dot(v1_rel, edge_dir)  # parameter along edge
                closest_on_edge = v1 + t * edge_dir
                
                # Only consider if closest point is "ahead" of observer (positive projection on ray)
                proj_dist = np.dot(closest_on_edge, ray_dir)
                
                if proj_dist < 0.1:  # Behind observer
                    continue
                
                # Perpendicular distance to ray
                perp_dist = np.linalg.norm(closest_on_edge - proj_dist * ray_dir)
                
                # Elevation angle to top of building at this point
                if proj_dist > 0:
                    elevation_angle = np.arctan2(height, proj_dist)
                    max_elevation = max(max_elevation, elevation_angle)
        
        max_elevation_angles.append(max_elevation)
    
    # Convert elevation angles to blocked fraction
    # Elevation 0° = horizon, 90° = zenith
    # Sky View Factor = (2π - Ω_blocked) / (2π) = 1 - (Ω_blocked / 2π)
    
    # Using solid angle of a cone: Ω = 2π(1 - cos(α))
    # But for discrete rays: blocked_rays / total_rays
    
    blocked_fraction = 0.0
    for elev_angle in max_elevation_angles:
        # Solid angle element for ray in this direction (assuming angular resolution)
        solid_angle_element = np.sin(angular_resolution_deg/2 * np.pi/180) * (2*np.pi / num_azimuth_sectors)
        # Elevation angle blocking: weight by sin(elevation) to account for projection
        blocked_fraction += np.sin(max(0, elev_angle)) * (1.0 / num_azimuth_sectors)
    
    svf = 1.0 - blocked_fraction
    svf = np.clip(svf, 0.0, 1.0)
    
    debug_info = {
        'num_buildings_considered': len(buildings_nearby),
        'max_elevation_angles_deg': [np.degrees(a) for a in max_elevation_angles],
        'mean_max_elevation_deg': np.degrees(np.mean(max_elevation_angles)),
        'max_elevation_deg': np.degrees(np.max(max_elevation_angles)),
    }
    
    return svf, debug_info


def calculate_urban_morphology_indices_physics_based(
    measurements_gdf, 
    buildings_gdf, 
    buffer_radius=100,
    use_physics_svf=True
):
    """
    Calculate urban morphology indices with optional physics-based SVF.
    
    Combines:
    - Simplified SVF (fast, uses area)
    - Physics-based SVF (slower, uses raycasting + solid angles)
    - Building height statistics
    - Density metrics
    
    Args:
        measurements_gdf: GeoDataFrame with measurement points
        buildings_gdf: GeoDataFrame with buildings
        buffer_radius: Radius in meters for local context
        use_physics_svf: If True, calculate physics-based SVF (slower but accurate)
    
    Returns:
        measurements_gdf with added columns
    """
    measurements_gdf["sky_view_factor"] = np.nan
    measurements_gdf["sky_view_factor_physics"] = np.nan
    measurements_gdf["building_surface_fraction"] = np.nan
    measurements_gdf["building_density"] = np.nan
    measurements_gdf["mean_building_height_in_buffer"] = np.nan
    measurements_gdf["aspect_ratio"] = np.nan  # H/W ratio
    
    n_points = len(measurements_gdf)
    
    for idx, point in measurements_gdf.iterrows():
        if (idx + 1) % max(1, n_points // 10) == 0:
            print(f"  Processing point {idx + 1}/{n_points}")
        
        # Create circular buffer
        buffer = point.geometry.buffer(buffer_radius / 111000)
        buildings_in_buffer = buildings_gdf[buildings_gdf.geometry.intersects(buffer)]
        
        if len(buildings_in_buffer) == 0:
            measurements_gdf.loc[idx, "sky_view_factor"] = 1.0
            if use_physics_svf:
                measurements_gdf.loc[idx, "sky_view_factor_physics"] = 1.0
            continue
        
        # === SIMPLIFIED SVF (area-based) ===
        building_area = buildings_in_buffer.geometry.area.sum()
        buffer_area = np.pi * (buffer_radius / 111000) ** 2
        measurements_gdf.loc[idx, "building_surface_fraction"] = building_area / buffer_area
        measurements_gdf.loc[idx, "sky_view_factor"] = max(0, 1 - (building_area / buffer_area))
        
        # === PHYSICS-BASED SVF (raycasting) ===
        if use_physics_svf:
            try:
                svf_physics, debug = calculate_svf_from_buildings_geometry(
                    point, 
                    buildings_in_buffer,
                    search_radius_m=buffer_radius,
                    num_azimuth_sectors=72
                )
                measurements_gdf.loc[idx, "sky_view_factor_physics"] = svf_physics
            except Exception as e:
                measurements_gdf.loc[idx, "sky_view_factor_physics"] = np.nan
        
        # === BUILDING STATISTICS ===
        heights = buildings_in_buffer["height_calc"].values
        measurements_gdf.loc[idx, "mean_building_height_in_buffer"] = np.nanmean(heights)
        measurements_gdf.loc[idx, "building_density"] = len(buildings_in_buffer) / ((buffer_radius**2) / 10000)
        
        # === ASPECT RATIO (H/W) ===
        # Approximate mean building height / average street width
        # Street width ≈ mean distance between buildings
        if len(heights) > 1:
            mean_height = np.nanmean(heights)
            # Estimate street width from building spacing
            perimeter = buffer.length
            building_count = len(buildings_in_buffer)
            estimated_street_width = (buffer_radius * 2) / (building_count + 1) if building_count > 0 else buffer_radius
            aspect_ratio = mean_height / estimated_street_width
            measurements_gdf.loc[idx, "aspect_ratio"] = aspect_ratio
    
    return measurements_gdf


#def calculate_urban_morphology_indices(
#    measurements_gdf, buildings_gdf, buffer_radius=100
#):
#    """
#    Wrapper for backward compatibility - calls physics-based version.
#    """
#    return calculate_urban_morphology_indices_physics_based(
#        measurements_gdf, buildings_gdf, buffer_radius, use_physics_svf=True
#    )

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
        "mean_5nn_building_height",
        "max_5nn_building_height",
        "building_density",
        "sky_view_factor",
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

    # TODO make heatmap interactive (choose the column for colorization)

    plugins.HeatMap(heat_data, radius=15, blur=25, max_zoom=1).add_to(m)

    # Add measurement points as markers
    for idx, row in valid_data.iterrows():
        color_intensity = (row[pollutant_col] - vmin) / (vmax - vmin)
        color_intensity = np.clip(color_intensity, 0, 1)

        # Color from blue (low) to red (high)
        color = f"hsl({int((1 - color_intensity) * 240)}, 100%, 50%)"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            popup=f"""Particles ufp: {row[pollutant_col]:.1f}
Particles : {row[""]:}

Mean Building height 5nn {row["mean_5nn_building_height"]:1f}
Building density: {row["building_density"]:1f}
""",
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


def create_density_heatmap_gridded(
    measurements_gdf,
    particle_col="ultrafine_ufp",
    grid_size=100,
    output_path="density_heatmap_grid.png",
):
    """
    Create gridded heatmap using spatial interpolation.
    """
    valid_data = measurements_gdf[measurements_gdf[particle_col].notna()].copy()

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


def analyze_street_canyons(measurements_gdf, buildings_gdf, output_dir="./results/"):
    """
    Identify and analyze potential street canyon effects.

    High pollution areas with high building height ratios indicate
    potential street canyon effects trapping particles.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Classify measurements by morphology
    q75 = measurements_gdf["nearest_building_height"].quantile(0.75)
    q25 = measurements_gdf["ultrafine_ufp"].quantile(0.25)
    q75_ufp = measurements_gdf["ultrafine_ufp"].quantile(0.75)

    street_canyon_mask = (measurements_gdf["nearest_building_height"] > q75) & (
        measurements_gdf["ultrafine_ufp"] > q75_ufp
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


def generate_analysis_report(measurements_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, output_dir="./results/"):
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

    #print("[3.0/6] Preparing SVF for raster"
    #import rvt
    #import rvt.vis
    # prepare rasterized "DEM"
    #dem = create_2d_dem_raster_from_3d_buildings(buildings_gdf)
    #out_dict= rvt.vis.sky_view_factor(
    #            dem,
    #            resolution=RASTER_RES,
    #            compute_svf=True,
    #            svf_r_max=10,
    #            svf_n_dir = 16,
    #        )
    #svf_arr = out_dict["svf"]



    # Step 3: Urban morphology
    print("[3.1/6] Calculating urban morphology indices...")
    measurements_gdf = calculate_urban_morphology_indices(
        measurements_gdf, buildings_gdf, 
    )


    print(f"Saving temp measurements gdf to {output_dir}")
    measurements_gdf.to_file(
        os.path.join(output_dir, "measurements_with_building_features_temp.gpkg")
    )

    exit("Exiting early")

    # Step 4: Statistical analysis
    print("[4/6] Performing correlation analysis...")
    results, analysis_df = perform_correlation_analysis(measurements_gdf, output_dir)

    # Step 5: Visualizations
    print("[5/6] Creating visualizations...")
    create_building_height_pollution_plot(
        measurements_gdf, os.path.join(output_dir, "building_height_vs_pollution.png")
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
    # Load data
    measurements = load_particle_measurements(sensor_position="particleFront")
    buildings = load_buildings()
    utm_epsg = 32632

    # **ADD THIS: Reproject to UTM if in lat/lon**
    if buildings.crs.to_string().startswith('EPSG:43'):  # lat/lon
        print(f"Converting buildings to epsg {utm_epsg}")

        # For Würzburg, use UTM zone 32N
        buildings = buildings.to_crs(epsg=utm_epsg)
        #point = gpd.GeoSeries([point], crs='EPSG:4326').to_crs(epsg=32632)[0]
    if measurements.crs.to_string().startswith('EPSG:43'):  # lat/lon
        # For Würzburg, use UTM zone 32N
        print(f"Converting measurements to epsg {utm_epsg}")
        measurements = measurements.to_crs(epsg=utm_epsg)
        #point = gpd.GeoSeries([point], crs='EPSG:4326').to_crs(epsg=32632)[0]
    
    # Run analysis
    analyzed_data, buildings_data = generate_analysis_report(measurements, buildings)
