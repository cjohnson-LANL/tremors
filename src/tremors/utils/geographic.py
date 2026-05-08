import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles
import matplotlib.patheffects as pe
from pyproj import Geod
from matplotlib.patches import Polygon
from cartopy.geodesic import Geodesic
import numpy as np


REGIONAL_RULES = {
    "NorthAmericaWest": {
        "bbox": (30, 70, -170, -115), 
        "dcs": ["IRIS", "SCEDC", "NCEDC", "EARTHSCOPE", "USGS", "IRISPH5"]
    },
    "NorthAmericaCentral": {
        "bbox": (25, 37, -107, -93),
        "dcs": ["TEXNET"]
    },
    "NewZealand": {
        "bbox": (-48, -34, 166, 179),
        "dcs": ["GEONET"]
    },
    "Canada": {
        "bbox": (42, 83, -141, -52),
        "dcs": ["NRCAN"]
    },
    "SouthAmerica": {
        "bbox": (-60, 15, -85, -30),
        "dcs": ["USP"]
    },
    "EuroMed": {
        "bbox": (30, 70, -25, 45),
        "dcs": ["INGV", "BGS", "RESIF", "NOA", "KOERI", "NIEP", "ETH", "GEOFON", "IPGP", "BGR", "EIDA", "EMSC", "GFZ", "ICGC", "KNMI", "LMU", "ODC", "ORFEUS", "RESIFPH5", "UIB-NORSAR"]
    },
    "Australia": {
        "bbox": (-45, -10, 110, 155),
        "dcs": ["AUSPASS"]
    },
    "Taiwan": {
        "bbox": (21, 26, 119, 123),
        "dcs": ["IESDMC"]
    },
    "Global": {
        "bbox": (-90, 90, -180, 180),
        "dcs": ["ISC", "RASPISHAKE", "EMSC"]
    }
}


def _convert_to_meters(distance, unit, ellipse='WGS84'):
    """Convert distance to meters based on unit."""
    unit = unit.lower()
    #rad adding in other possiblilities we errored out here we might want to regex this
    if unit in ['km','kilometer','kilometers', 'kilometre','kilometres']: 
        return distance * 1000
    elif unit in ['m','meter','meters','metre','metres']:
        return distance
    elif unit in ['deg','degree','degrees']:
        g = Geod(ellps=ellipse)
        _, _, dist_m = g.inv(0, 0, 0, distance)
        return abs(dist_m)
    else:
        raise ValueError(f"Unsupported unit '{unit}'. Must be one of: 'km', 'm', 'degrees'")


def boundingbox(lat, lon, distance, unit='km', ellipse='WGS84'):
    """Calculate a bounding box around a point given distance and ellipsoid.
    
    This function computes a square bounding box around a central point defined by
    latitude and longitude. The box is calculated by finding four corner points at
    equal distance from the center point along diagonal directions (45°, 135°, 225°, 315°),
    taking into account the Earth's ellipsoidal shape.
    
    Args:
        lat: Latitude of the center point in decimal degrees (-90 to 90)
        lon: Longitude of the center point in decimal degrees (-180 to 180)
        distance: Distance from center point to corners (must be positive)
        unit: Unit of the distance parameter. One of:
              'km' (default) - kilometers
              'm'            - meters
              'degrees'      - decimal degrees (converted via pyproj/WGS84)
        ellipse: Name of the ellipsoid to use for geodetic calculations.
                 Must be one of 'WGS84' (default), 'clrk66', or 'GRS80'
    
    Returns:
        Tuple containing (latmin, latmax, lonmin, lonmax) where:
            latmin: Minimum latitude of the bounding box
            latmax: Maximum latitude of the bounding box
            lonmin: Minimum longitude of the bounding box
            lonmax: Maximum longitude of the bounding box
    """
    # Input validation
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if distance <= 0:
        raise ValueError("Distance must be positive")
    if ellipse not in ['WGS84', 'clrk66', 'GRS80']:
        raise ValueError("Unsupported ellipsoid. Must be one of: WGS84, clrk66, GRS80")

    # Convert distance to meters using pyproj
    distance_m = _convert_to_meters(distance, unit)

    # Create geodesic calculation object
    g = Geod(ellps=ellipse)

    # Calculate corner points using forward geodetic calculation
    # Returns tuples of (lon, lat, back azimuth) for each corner
    corners = [
        g.fwd(lon, lat, azimuth, distance_m)
        for azimuth in [315, 45, 135, 225]  # Counter-clockwise from top-left
    ]

    # Convert corners to numpy array and remove back azimuth column
    corner_points = np.array(corners)[:, :2]

    # Extract min/max coordinates
    lonmin, lonmax = np.min(corner_points[:, 0]), np.max(corner_points[:, 0])
    latmin, latmax = np.min(corner_points[:, 1]), np.max(corner_points[:, 1])

    return latmin, latmax, lonmin, lonmax


def boundingradius(lat, lon, distance, unit='km', numpoints=361, ellipse='WGS84'):
    """Calculate a circular bounding polygon around a point.

    Args:
        lat: Latitude of the center point in decimal degrees (-90 to 90)
        lon: Longitude of the center point in decimal degrees (-180 to 180)
        distance: Radius distance from center point (must be positive)
        unit: Unit of the distance parameter. One of:
              'km' (default) - kilometers
              'm'            - meters
              'degrees'      - decimal degrees (converted via pyproj/WGS84)
        numpoints: Number of points to generate around the circle (default 361)
        ellipse: Name of the ellipsoid to use for geodetic calculations.
                 Must be one of 'WGS84' (default), 'clrk66', or 'GRS80'

    Returns:
        numpy array of shape (numpoints, 2) containing [lat, lon] pairs
    """
    # Input validation
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if distance <= 0:
        raise ValueError("Distance must be positive")
    if ellipse not in ['WGS84', 'clrk66', 'GRS80']:
        raise ValueError("Unsupported ellipsoid. Must be one of: WGS84, clrk66, GRS80")

    # Convert distance to meters
    distance_m = _convert_to_meters(distance, unit)

    g = Geod(ellps=ellipse)

    points = np.zeros((numpoints, 2))
    angles = np.linspace(0, 360, numpoints)

    for i, angle in enumerate(angles):
        point_lon, point_lat, _ = g.fwd(lon, lat, angle, distance_m)
        points[i] = [point_lat, point_lon]

    return points

def add_north_arrow(ax, x=0.96, y=0.03, length=0.08, fontsize=16):
    ann = ax.annotate(
        'N', xy=(x, y + length), xytext=(x, y), xycoords='axes fraction',
        textcoords='axes fraction', ha='center', va='center', fontsize=fontsize,
        fontweight='bold', color='k', arrowprops=dict(arrowstyle='-|>', facecolor='k', edgecolor='k', linewidth=2, shrinkA=0, shrinkB=0), zorder=30)
    ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

def add_scalebar(ax, length_km=25, location=(0.04, 0.035), linewidth=3, fontsize=14):
    geod = Geodesic()
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())
    x_frac, y_frac = location
    lon0 = lon_min + x_frac * (lon_max - lon_min)
    lat0 = lat_min + y_frac * (lat_max - lat_min)
    end = geod.direct(points=np.array([[lon0, lat0]]), azimuths=np.array([90.0]), distances=np.array([length_km * 1000.0]))
    lon1, lat1 = end[0, 0], end[0, 1]
    tick_h = 0.01 * (lat_max - lat_min)
    ax.plot([lon0, lon1], [lat0, lat1], transform=ccrs.PlateCarree(), color='k', linewidth=linewidth*1.25, solid_capstyle='butt', zorder=30)
    ax.plot([lon0, lon0], [lat0 - tick_h, lat0 + tick_h], transform=ccrs.PlateCarree(), color='k', linewidth=linewidth/1.5, zorder=30)
    ax.plot([lon1, lon1], [lat1 - tick_h, lat1 + tick_h], transform=ccrs.PlateCarree(), color='k', linewidth=linewidth/1.5, zorder=30)
    lon_mid, lat_mid = 0.5 * (lon0 + lon1), 0.5 * (lat0 + lat1)
    txt = ax.text(lon_mid, lat_mid - 4.5 * tick_h, f'{length_km:g} km', transform=ccrs.PlateCarree(), ha='center', va='bottom', fontsize=fontsize, color='k', zorder=31)
    txt.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])
