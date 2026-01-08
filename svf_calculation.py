"""
Sky View Factor (SVF) Calculation using Formula 7 from:
"Online Street View-Based Approach for Sky View Factor Estimation: A Case Study of Nanjing, China"
https://www.mdpi.com/2076-3417/14/5/2133

This module provides functions to calculate SVF for a given point based on OSM building data
with height information.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from typing import Tuple, Union


def calculate_svf(
    point: Union[Tuple[float, float], Point],
    building_gdf: gpd.GeoDataFrame,
    observer_height: float = 0,
    num_rings: int = 36,
    max_distance: float = 100.0,
    azimuth_divisions: int = 360,
) -> float:
    """
    Calculate Sky View Factor (SVF) for a given point based on building geometries and heights.

    This function implements Formula 7 from the paper, which calculates SVF as:
    SVF_PA = (1 / (2π sin(π/2n))) * Σ sin((2i-1)π/2n) * α_i

    Parameters
    ----------
    point : tuple or shapely.geometry.Point
        The observation point as (x, y) or Point object (in the same CRS as building_gdf)
    building_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons with a 'height' column (in meters)
        Heights should be positive numeric values
    observer_height : float, default=1.7
        Height of observer above ground level (in meters)
    num_rings : int, default=36
        Number of radial divisions for SVF calculation. Higher values = more accuracy
        but slower computation. Default 36 corresponds to 10° divisions
    max_distance : float, default=500.0
        Maximum distance to consider buildings (in meters). Buildings beyond this
        distance are ignored
    azimuth_divisions : int, default=360
        Number of azimuth divisions (angular directions). Default 360 = 1° divisions

    Returns
    -------
    float
        Sky View Factor value between 0 and 1
        - SVF = 1.0 means completely open sky (no obstructions)
        - SVF = 0.0 means completely blocked sky

    Examples
    --------
    >>> import geopandas as gpd
    >>> # Load your building data
    >>> buildings = gpd.read_file('buildings.gpkg')
    >>> point = (10.5, 20.3)  # observation point
    >>> svf = calculate_svf(point, buildings)
    >>> print(f"SVF: {svf:.3f}")
    SVF: 0.645

    Notes
    -----
    - The building_gdf must have a 'height' column with positive numeric values
    - Point coordinates and building coordinates must be in the same CRS
    - For best results, use projected coordinate systems (e.g., UTM) rather than lat/lon
    - Computation time increases with number of buildings and num_rings
    """
    # Convert point to Point object if tuple
    if isinstance(point, tuple):
        point = Point(point)

    # Filter buildings within max_distance
    buildings_filtered = building_gdf[
        building_gdf.geometry.distance(point) <= max_distance
    ].copy()

    if len(buildings_filtered) == 0:
        # No buildings nearby, completely open sky
        return 1.0

    # Initialize arrays for angular calculations
    azimuths = np.linspace(0, 2 * np.pi, azimuth_divisions, endpoint=False)
    azimuth_step = 2 * np.pi / azimuth_divisions

    # Initialize array to store maximum elevation angles for each azimuth
    max_elevation_angles = np.zeros(azimuth_divisions)

    # Calculate maximum elevation angle for each azimuth direction
    for building_idx, (_, building) in enumerate(buildings_filtered.iterrows()):
        building_geom = building.geometry
        building_height = building["height_calc"]

        if not isinstance(building_geom, Polygon):
            continue

        # Get building vertices # TODO -> fix for all geoms
        building_coords = list(building_geom.exterior.coords[:-1])

        # For each vertex, calculate the elevation angle and azimuth
        for coord in building_coords:
            vertex = Point(coord)
            dx = vertex.x - point.x
            dy = vertex.y - point.y
            horizontal_distance = np.sqrt(dx**2 + dy**2)

            if horizontal_distance < 0.1:
                # Point is inside or very close to building
                max_elevation_angles[:] = np.pi / 2
                continue

            # Calculate azimuth angle (0 to 2π)
            azimuth = np.arctan2(dy, dx)
            if azimuth < 0: # why not simply + np.pi
                azimuth += 2 * np.pi

            # Height of building top above observer
            height_above_observer = building_height - observer_height

            # Calculate elevation angle to top of building
            elevation_angle = np.arctan(height_above_observer / horizontal_distance)

            if elevation_angle < 0:
                elevation_angle = 0

            # Find closest azimuth bin and update if this is higher
            azimuth_idx = int(np.round(azimuth / azimuth_step)) % azimuth_divisions
            max_elevation_angles[azimuth_idx] = max(
                max_elevation_angles[azimuth_idx], elevation_angle
            )

    # Calculate SVF using Formula 7
    # SVF = (1 / (2π sin(π/2n))) * Σ sin((2i-1)π/2n) * α_i
    n = num_rings
    normalization_factor = 1.0 / (2 * np.pi * np.sin(np.pi / (2 * n)))

    svf = 0.0
    for i in range(1, n + 1):
        # Weight for this ring
        weight = np.sin((2 * i - 1) * np.pi / (2 * n))

        # Calculate average sky visibility in this ring
        # Rings are defined by elevation angle ranges
        ring_start_angle = (i - 1) * np.pi / (2 * n)
        ring_end_angle = i * np.pi / (2 * n)

        # For each azimuth, check if sky is visible in this ring
        sky_visible_count = 0
        for az_idx, max_elev in enumerate(max_elevation_angles):
            # Middle of the elevation range for this ring
            mid_ring_angle = (ring_start_angle + ring_end_angle) / 2

            # Sky is visible if building doesn't block this elevation
            if max_elev < mid_ring_angle:
                sky_visible_count += 1

        # Fraction of azimuths with visible sky in this ring
        visible_fraction = sky_visible_count / azimuth_divisions

        # Add contribution from this ring
        svf += weight * visible_fraction

    # Apply normalization
    svf *= normalization_factor

    # Clamp to [0, 1]
    svf = np.clip(svf, 0.0, 1.0)

    return svf



def calculate_svf_easy(
    point: Union[Tuple[float, float], Point],
    building_gdf: gpd.GeoDataFrame,
    observer_height: float = 0,
    num_rings: int = 36,
    max_distance: float = 100.0,
    azimuth_divisions: int = 360,
) -> float:
    """
    Calculate Sky View Factor (SVF) for a given point based on building geometries and heights.

    This function implements Formula 7 from the paper, which calculates SVF as:
    SVF_PA = (1 / (2π sin(π/2n))) * Σ sin((2i-1)π/2n) * α_i

    Parameters
    ----------
    point : tuple or shapely.geometry.Point
        The observation point as (x, y) or Point object (in the same CRS as building_gdf)
    building_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons with a 'height' column (in meters)
        Heights should be positive numeric values
    observer_height : float, default=1.7
        Height of observer above ground level (in meters)
    num_rings : int, default=36
        Number of radial divisions for SVF calculation. Higher values = more accuracy
        but slower computation. Default 36 corresponds to 10° divisions
    max_distance : float, default=500.0
        Maximum distance to consider buildings (in meters). Buildings beyond this
        distance are ignored
    azimuth_divisions : int, default=360
        Number of azimuth divisions (angular directions). Default 360 = 1° divisions

    Returns
    -------
    float
        Sky View Factor value between 0 and 1
        - SVF = 1.0 means completely open sky (no obstructions)
        - SVF = 0.0 means completely blocked sky

    Examples
    --------
    >>> import geopandas as gpd
    >>> # Load your building data
    >>> buildings = gpd.read_file('buildings.gpkg')
    >>> point = (10.5, 20.3)  # observation point
    >>> svf = calculate_svf(point, buildings)
    >>> print(f"SVF: {svf:.3f}")
    SVF: 0.645

    Notes
    -----
    - The building_gdf must have a 'height' column with positive numeric values
    - Point coordinates and building coordinates must be in the same CRS
    - For best results, use projected coordinate systems (e.g., UTM) rather than lat/lon
    - Computation time increases with number of buildings and num_rings
    """
    # Convert point to Point object if tuple
    if isinstance(point, tuple):
        point = Point(point)

    # Filter buildings within max_distance
    buildings_filtered = building_gdf[
        building_gdf.geometry.distance(point) <= max_distance
    ].copy()

    if len(buildings_filtered) == 0:
        # No buildings nearby, completely open sky
        return 1.0

    # Initialize arrays for angular calculations
    azimuths = np.linspace(0, 2 * np.pi, azimuth_divisions, endpoint=False)
    azimuth_step = 2 * np.pi / azimuth_divisions

    # Initialize array to store maximum elevation angles for each azimuth
    max_elevation_angles = np.zeros(azimuth_divisions)

    # Calculate maximum elevation angle for each azimuth direction

    for azimuth in azimuths:
        # cast ray and find building with maximum angle
        max_elev_for_azimuth = 0.0


        for building_idx, (_, building) in enumerate(buildings_filtered.iterrows()):
            building_geom = building.geometry
            building_height = building["height_calc"]

            if not isinstance(building_geom, Polygon):
                continue

            # TODO implement for multipolygons

            building_coords = list(building_geom.exterior.coords[:-1])

            for coord in building_coords:
                vertex = Point(coord)
                dx = vertex.x - point.x
                dy = vertex.y - point.y
                horizontal_distance = np.sqrt(dx**2 + dy**2)

                if horizontal_distance < 0.1:
                    max_elev_for_azimuth = np.pi / 2
                    break

                vertex_azimuth = np.arctan2(dy, dx)
                if vertex_azimuth < 0:
                    vertex_azimuth += 2 * np.pi

                azimuth_diff = abs(vertex_azimuth - azimuth)
                if azimuth_diff > np.pi:
                    azimuth_diff = 2 * np.pi - azimuth_diff

                if azimuth_diff < azimuth_step / 2:
                    height_above_observer = building_height - observer_height
                    elevation_angle = np.arctan(height_above_observer / horizontal_distance)
                    max_elev_for_azimuth = max(max_elev_for_azimuth, elevation_angle, 0)

        azimuth_idx = int(np.round(azimuth / azimuth_step)) % azimuth_divisions
        max_elevation_angles[azimuth_idx] = max_elev_for_azimuth

    # NEW
    # calculate svf for each azimuth division
    # based on that arctan returns values from [0, pi]
    # get the porporional of max_elevation_angle to pi
    # at 0° -> 100% weight, at 90° -> 0% weight
    print("max elev angles:", max_elevation_angles)
    # cos as weight
    weighted_max_elev_angles = [np.cos(a) * a for a in max_elevation_angles]
    # weighted_max_elev_angles = [a for a in max_elevation_angles]

    print("weighted: ", weighted_max_elev_angles)

    svf_easy = sum(weighted_max_elev_angles) / (np.pi/2 * azimuth_divisions)
    return svf_easy


def calculate_svf_easy_raycasting(
    point: Union[Tuple[float, float], Point],
    building_gdf: gpd.GeoDataFrame,
    observer_height: float = 0,
    max_distance: float = 100.0,
    azimuth_divisions: int = 360,
) -> float:
    """
    Calculate Sky View Factor (SVF) for a given point based on building geometries and heights.

    This function implements a variation of Formula 7 from the paper, which calculates SVF as:
    SVF_PA = (1 / (2π sin(π/2n))) * Σ sin((2i-1)π/2n) * α_i

    Parameters
    ----------
    point : tuple or shapely.geometry.Point
        The observation point as (x, y) or Point object (in the same CRS as building_gdf)
    building_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons with a 'height' column (in meters)
        Heights should be positive numeric values
    observer_height : float, default=1.7
        Height of observer above ground level (in meters)
    max_distance : float, default=500.0
        Maximum distance to consider buildings (in meters). Buildings beyond this
        distance are ignored
    azimuth_divisions : int, default=360
        Number of azimuth divisions (angular directions). Default 360 = 1° divisions

    Returns
    -------
    float
        Sky View Factor value between 0 and 1
        - SVF = 1.0 means completely open sky (no obstructions)
        - SVF = 0.0 means completely blocked sky

    Examples
    --------
    >>> import geopandas as gpd
    >>> # Load your building data
    >>> buildings = gpd.read_file('buildings.gpkg')
    >>> point = (10.5, 20.3)  # observation point
    >>> svf = calculate_svf(point, buildings)
    >>> print(f"SVF: {svf:.3f}")
    SVF: 0.645

    Notes
    -----
    - The building_gdf must have a 'height' column with positive numeric values
    - Point coordinates and building coordinates must be in the same CRS
    - For best results, use projected coordinate systems (e.g., UTM) rather than lat/lon
    - Computation time increases with number of buildings and num_rings
    """
    # Convert point to Point object if tuple
    if isinstance(point, tuple):
        point = Point(point)

    # in building
    if len(building_gdf[building_gdf.geometry.distance(point) == 0]):
        return 0.0

    # Filter buildings within max_distance
    buildings_filtered = building_gdf[
        building_gdf.geometry.distance(point) <= max_distance
    ].copy()

    if len(buildings_filtered) == 0:
        # No buildings nearby, completely open sky
        return 1.0

    # Initialize arrays for angular calculations
    azimuths = np.linspace(0, 2 * np.pi, azimuth_divisions, endpoint=False)
    azimuth_step = 2 * np.pi / azimuth_divisions

    # Initialize array to store maximum elevation angles for each azimuth
    max_elevation_angles = np.zeros(azimuth_divisions)

    # Calculate maximum elevation angle for each azimuth direction

    for azimuth in azimuths:
        # cast ray and find building with maximum angle
        max_elev_for_azimuth = 0.0

        # create ray
        ray_end = Point(point.x + max_distance * np.cos(azimuth), 
                                  point.y + max_distance * np.sin(azimuth))
        
        ray = LineString([point, ray_end])

        # For all buildings intersected by the ray in this azimuth direction,
        # calculate the maximum elevation angle and update the result
        azimuth_idx = int(np.round(azimuth / azimuth_step)) % azimuth_divisions

        for building_idx, (_, building) in enumerate(buildings_filtered.iterrows()):
            building_geom = building.geometry
            building_height = building["height_calc"]

            if not isinstance(building_geom, Polygon):
                continue

            # TODO implement for multipolygons

            building_coords = list(building_geom.exterior.coords[:-1])

            # Find intersection points
            intersection = building_geom.intersection(ray)
            if not intersection.is_empty:
                if intersection.geom_type == 'Point':
                    intersection_points = [intersection]
                elif intersection.geom_type == 'LineString':
                    intersection_points = [Point(intersection.coords[0]), Point(intersection.coords[-1])]
                elif intersection.geom_type == 'MultiPoint':
                    intersection_points = list(intersection.geoms)
                else:
                    intersection_points = []
                
                # probably redundant, just find nearest point and get max elev for that one
                for int_point in intersection_points:
                    dx = int_point.x - point.x
                    dy = int_point.y - point.y
                    horizontal_distance = np.sqrt(dx**2 + dy**2)
                    
                    #if horizontal_distance < 0.1:
                    #    max_elev_for_azimuth = np.pi / 2
                    #    break
                    
                    height_above_observer = building_height - observer_height
                    elevation_angle = np.arctan(height_above_observer / horizontal_distance)
                    max_elev_for_azimuth = max(max_elev_for_azimuth, elevation_angle, 0)
            
        azimuth_idx = int(np.round(azimuth / azimuth_step)) % azimuth_divisions
        max_elevation_angles[azimuth_idx] = max_elev_for_azimuth

    # NEW
    # calculate svf for each azimuth division
    # based on that arctan returns values from [0, pi]
    # get the porporional of max_elevation_angle to pi
    # at 0° -> 100% weight, at 90° -> 0% weight ??
    print("max elev angles:", max_elevation_angles)
    weighted_max_elev_angles = [np.cos(a) * a for a in max_elevation_angles]
    weighted_max_elev_angles = [a for a in max_elevation_angles]

    
    # cos as weight
    # weighted_max_elev_angles = [cos(a) * a for a in max_elevation_angles]

    print("weighted: ", weighted_max_elev_angles)

    full_view = (np.pi/2 * azimuth_divisions)
    svf_easy = 1 - (sum(weighted_max_elev_angles) / full_view)
    return svf_easy


def calculate_svf_batch(
    points: list,
    building_gdf: gpd.GeoDataFrame,
    observer_height: float = 1.7,
    num_rings: int = 36,
    max_distance: float = 500.0,
    azimuth_divisions: int = 360,
    verbose: bool = False,
) -> list:
    """
    Calculate SVF for multiple points.

    Parameters
    ----------
    points : list of tuple or Point
        List of observation points
    building_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building data
    observer_height : float
        Height of observer (default 1.7 m)
    num_rings : int
        Number of radial divisions (default 36)
    max_distance : float
        Maximum distance to consider buildings (default 500 m)
    azimuth_divisions : int
        Number of azimuth divisions (default 360)
    verbose : bool
        If True, print progress information

    Returns
    -------
    list
        List of SVF values corresponding to input points
    """
    svf_values = []

    for idx, point in enumerate(points):
        if verbose and (idx + 1) % 100 == 0:
            print(f"Calculating SVF for point {idx + 1}/{len(points)}")

        svf = calculate_svf_easy_raycasting(
            point,
            building_gdf,
            observer_height=observer_height,
            max_distance=max_distance,
            azimuth_divisions=azimuth_divisions,
        )
        svf_values.append(svf)

    return svf_values


if __name__ == "__main__":
    # Example usage
    print("SVF Calculation Module")
    print("=" * 50)
    print(
        "This module calculates Sky View Factor using the method from:\n"
        "Xu et al. (2024) - Online Street View-Based Approach for SVF Estimation"
    )
    print(
        "\nUsage:\n"
        "  from svf_calculation import calculate_svf\n"
        "  svf = calculate_svf(point, buildings_gdf)\n"
    )
