import numpy
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from geojson import Feature, LineString, FeatureCollection

grid_size = 1.0
latrange = numpy.arange(-90.0, 90.0, grid_size)
lonrange = numpy.arange(-180.0, 180.0, grid_size)
X, Y = numpy.meshgrid(lonrange, latrange)
Z = numpy.sqrt(X * X + Y * Y)

figure = plt.figure()
ax = figure.add_subplot(111)
contour = ax.contour(lonrange, latrange, Z, levels=numpy.linspace(start=0, stop=100, num=10), cmap=plt.cm.jet)

line_features = []
paths = contour.get_paths()
color = contour.get_edgecolor()
for path in paths:
    v = path.vertices
    coordinates = []
    for i in range(len(v)):
        lat = v[i][0]
        lon = v[i][1]
        coordinates.append((lat, lon))
    line = LineString(coordinates)
    properties = {
        "stroke-width": 3,
        "stroke": rgb2hex(color[0]),
    }
    line_features.append(Feature(geometry=line, properties=properties))

feature_collection = FeatureCollection(line_features)
geojson_dump = geojson.dumps(feature_collection, sort_keys=True)
with open('contour.geojson', 'w') as fileout:
    fileout.write(geojson_dump)