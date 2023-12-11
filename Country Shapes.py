# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:31:49 2023

@author: jkora
"""

from geopy.geocoders import Nominatim

import numpy as np

import shapefile

shape = shapefile.Reader("TM_WORLD_BORDERS-0.3.dbf")
print(shape)
feature = shape.shapeRecords()[0]
first = feature.shape.__geo_interface__

print(first)
'''
geolocator = Nominatim()

def geolocate(country):
    try:
        loc = geolocator.geocode(country)
    except:
        return np.nan

print(geolocate("United States"))
'''