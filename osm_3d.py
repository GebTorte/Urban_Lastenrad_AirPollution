import geopandas as gpd
from sqlalchemy import create_engine
import pydeck as pdk
import pandas as pd
import os

# --- Configuration ---
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'your_database_name'
TABLE_NAME = 'buildings'

# The center of your map (adjust as needed for the location of your buildings)
# Example for a general Würzburg area
VIEW_LATITUDE = 49.7913
VIEW_LONGITUDE = 9.9536

# default height in m
# maybe cacl mean for aoi
DEFAULT_BUILDING_HEIGHT = 12

# --- Database Connection ---
# Construct the database URL
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

print(f"Connecting to database: {DB_NAME}...")

# --- Data Retrieval ---
try:
    # SQL query to select data and transform geometry to WGS 84 (EPSG:4326) 
    # for compatibility with web mapping libraries like Pydeck.
    sql_query = f"""
    SELECT 
        id, 
        height, 
        ST_AsText(ST_Transform(geom, 4326)) AS geom_wkt
    FROM 
        {TABLE_NAME}
    WHERE
        height IS NOT NULL AND ST_IsValid(geom)
    """
    
    # Read the data into a Pandas DataFrame first to handle WKT conversion
    df = pd.read_sql_query(sql_query, engine)

    # Convert WKT to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geom_wkt']), crs="EPSG:4326")

    print(f"Retrieved {len(gdf)} buildings.")

except Exception as e:
    print(f"Error connecting to the database or retrieving data: {e}")
    exit()

# Ensure height is numeric and handle potential issues
gdf['height'] = pd.to_numeric(gdf['height'], errors='coerce').fillna(DEFAULT_BUILDING_HEIGHT)

# Prepare data for Pydeck (needs a standard DataFrame with latitude and longitude columns)
# Extract centroids for positioning the 3D columns/buildings if needed, or use the polygon layer directly
# The ColumnLayer/PolygonLayer in Pydeck uses the geometry directly.

# Convert GeoDataFrame to a regular pandas DataFrame for Pydeck, ensuring 'height' is present
# Pydeck's PolygonLayer handles the geometry column directly if it is in the correct format (e.g., Shapely objects).
# Pydeck expects coordinates in list format.

# For Pydeck's PolygonLayer, we just need the gdf in 4326 CRS.

# --- 3D Visualization using Pydeck ---

# Define the 3D building layer
building_layer = pdk.Layer(
    'PolygonLayer',
    data=gdf,
    get_polygon='geometry.exterior.coords', # This is where we use the geopandas geometry
    get_elevation='height',
    elevation_scale=1,
    get_fill_color=[200, 200, 255, 180], # Light blue buildings
    pickable=True,
    auto_highlight=True,
    extruded=True
)

# Set the view state with the initial camera position and pitch for a 3D effect
view_state = pdk.ViewState(
    latitude=VIEW_LATITUDE,
    longitude=VIEW_LONGITUDE,
    zoom=14,
    pitch=45,
    bearing=0
)

# Create the Pydeck map
r = pdk.Deck(
    layers=[building_layer],
    initial_view_state=view_state,
    tooltip={"text": "Building Height: {height} meters"},
    map_style='mapbox://styles/mapbox/light-v10' # Requires Mapbox token for non-default styles
)

# You may need a Mapbox API token for non-default map styles. 
# Set it as an environment variable (e.g., MAPBOX_API_KEY) or use a free style like:
# map_style='basemaps.cartocdn.com'

# Save the interactive map to an HTML file
output_path = "interactive_3d_map.html"
r.to_html(output_path)

print(f"Interactive 3D map saved to {os.path.abspath(output_path)}")
print("Open the HTML file in your web browser to view the map.")
