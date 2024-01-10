import plotly.express as px
import pandas as pd
import numpy as np
from numpy import pi,sin, cos, sqrt, arctan2

from plotly.offline import plot
import plotly.graph_objects as go

from shapely import MultiPolygon, Polygon, affinity, wkt
from shapely.ops import transform
import shapely.geometry as sg
import shapely.ops as so


import geopandas as gpd

import matplotlib.pyplot as plt

filename = "file.txt"
#read in file of all countries' data
world = pd.read_csv(filename)

#read in hat outline csv
hat = pd.read_csv("T:\Hat_Outline\Hat_outline.csv")

#find area of the hat
hat_raw_poly = Polygon(list(zip(hat['x'],hat['y'])))
hat_area = hat_raw_poly.area


#normalize hat area to 1
hat['x_norm'] = hat['x']*np.sqrt(1/(hat_area))
hat['y_norm'] = hat['y']*np.sqrt(1/(hat_area))

hat_norm_poly = Polygon(list(zip(hat['x_norm'],hat['y_norm'])))
hat_center = np.array(hat_norm_poly.centroid.coords)[0]

#Check it is normalized
#print(PolyArea(hat['x_norm'],hat['y_norm']))

hat['x_cent'] = hat['x_norm'] - hat_center[0]
hat['y_cent'] = hat['y_norm'] - hat_center[1]

#make list of tuples from dataframe
column1 = 'x_cent'
column2 = 'y_cent'
hat_list = list(zip(hat[column1], hat[column2]))

hat_polygon = Polygon(hat[[column1,column2]])

#rotate hat    
#hat_polygon = affinity.rotate(hat_polygon, 70) #Canada
#hat_polygon = affinity.rotate(hat_polygon, 75) #Russia
#hat_polygon = affinity.rotate(hat_polygon, 120) #Mexico

#flip Polygon
#hat_polygon = transform(lambda x, y, z=None: (x, -y), hat_polygon) #Mexico

#--------------------------------------------------------------------------------
def norm_cent_poly(poly, country = None):
    c_area = poly.area

    blobs = []
    if poly.geom_type == 'MultiPolygon':
        for i in range(len(poly.geoms)):
            x,y = poly.geoms[i].exterior.xy
            x_corr = np.array(x)
            y_corr = np.array(y)
            if country == "Russia":
                x_corr = np.where(x_corr < 0, x_corr+360, x_corr)
            
            cx_norm = x_corr*np.sqrt(1/(c_area))
            cy_norm = y_corr*np.sqrt(1/(c_area))

            blobs.append(Polygon(list(zip(cx_norm, cy_norm))))

        country_norm = MultiPolygon(blobs)

        cent = np.array(country_norm.centroid.coords)[0]

        blobs = []

        for i in range(len(country_norm.geoms)):
            x,y = country_norm.geoms[i].exterior.xy
            x = np.array(x)
            y = np.array(y)

            cx_cent = x - cent[0]
            cy_cent = y - cent[1]

            blobs.append(Polygon(list(zip(cx_cent, cy_cent))))

        country_blob = MultiPolygon(blobs)

    elif poly.geom_type == 'Polygon':
        x,y = poly.exterior.xy
        x_corr = np.array(x)
        y_corr = np.array(y)
            
        cx_norm = x_corr*np.sqrt(1/(c_area))
        cy_norm = y_corr*np.sqrt(1/(c_area))

        country_norm = Polygon(list(zip(cx_norm, cy_norm)))

        cent = np.array(country_norm.centroid.coords)[0]

        x,y = country_norm.exterior.xy
        x = np.array(x)
        y = np.array(y)

        cx_cent = x - cent[0]
        cy_cent = y - cent[1]

        country_blob = Polygon(list(zip(cx_cent, cy_cent)))

    return country_blob

#-------------------------------------------------------------------------

def find_similarity_score(hat_polygon, country_poly, show = False):

    unique_hat_poly = hat_polygon.difference(country_poly)

    similarity_score = 100*(1-unique_hat_poly.area)

    if show is True:
        print("Normalized Hat Area {0:,.2f}".format(hat_polygon.area))
        print("Normalized Country Area {0:,.2f}".format(country_poly.area))
        print("Similarity Score: {0:.2f}%".format(similarity_score))
    
    elif show is False:
        pass
    
    return unique_hat_poly, similarity_score

#-------------------------------------------------------------------------

#plot the polygon
def plot_poly(poly, shape = "exterior", color = 'green', name = None):
    if poly.geom_type == 'MultiPolygon':
        for geom in list(poly.geoms):
            if shape == "exterior":
                xs, ys = geom.exterior.xy

                fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys),
                                        mode = "lines", 
                                        fill = 'toself', fillcolor=color,
                                        line = {'color' : color}, name = name))
            elif shape == "interior":
                for inner in geom.interiors:
                    xs, ys = inner.xy

                    fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys),
                                            mode = "lines", showlegend = False,
                                            fill = 'toself', fillcolor=color,
                                            line = {'color' : color}, name = name))
            else:
                print("Please define exterior or interior")
    elif poly.geom_type == 'Polygon':
        if shape == "exterior":
            xs, ys = poly.exterior.xy

            fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys),
                                        mode = "lines", 
                                        fill = 'toself', fillcolor=color,
                                        line = {'color' : color}, name = name))
            
        elif shape == "interior":
            xs, ys = inner.xy

            fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys),
                                    mode = "lines", showlegend = False,
                                    fill = 'toself', fillcolor=color,
                                    line = {'color' : color}, name = name))
        else:
            print("Please define exterior or interior")
    else:
        raise IOError('Shape is not a shapely polygon or multipolygon.')
    
#--------------------------------------------------------------------------------

world['Similarity Score'] = 0
world['Subtracted Hat Poly'] = 0
world['Norm Cent Country'] = 0
world['Rotation'] = 0
world['Flip'] = 0

similarity_dict = {}
country_list = world['name'].tolist()
#hat_polygon = transform(lambda x, y, z=None: (x, -y), hat_polygon)

for country in country_list:
    c = world[(world.name == country)]['geometry'].item()
    c = wkt.loads(c)
    country_poly = norm_cent_poly(c, country == country)
    #rotate the hat every 10 degrees and check overlap
    best = [0,0,0,0] #[unique_hat_poly, similarity_score, i*10, flip]
    for i in range(0,360,5):
        hat_polygon = affinity.rotate(hat_polygon, i)
        for flip in [True,False]:
            if flip is True:
                hat_polygon = transform(lambda x, y, z=None: (x, -y), hat_polygon)
            elif flip is False:
                pass
            else:
                print("error, flip neither True or False")

        unique_hat_poly, similarity_score = find_similarity_score(hat_polygon,
                                                                country_poly,
                                                                show = False)
        
        if similarity_score > best[1]:
            best = [unique_hat_poly, similarity_score, i, flip]
        else:
            pass
    
    world.loc[(world.name == country),
                'Norm Cent Country'] = country_poly
    world.loc[(world.name == country),
                'Subtracted Hat Poly'] = best[0] #unique_hat_poly
    world.loc[(world.name == country),
                'Similarity Score'] = best[1] #similarity_score
    world.loc[(world.name == country),
                'Rotation'] = best[2] #hat rotation
    world.loc[(world.name == country),
                'Flip'] = best[3] #was the hat flipped

#unique_hat_poly, similarity_score = find_similarity_score(hat_polygon, country_poly)

print(world.loc[world['Similarity Score'].idxmax()])

country = world.loc[world['Similarity Score'].idxmax(), 'name']

#unique_hat_poly = wkt.loads(world[(world.name == country)]['Subtracted Hat Poly'].item())
unique_hat_poly = world[(world.name == country)]['Subtracted Hat Poly'].item()
country_poly = world[(world.name == country)]['Norm Cent Country'].item()

world.to_csv("world_similarity.csv")

#-----------------------------------------------------------------------------

fig = go.Figure().update_layout(plot_bgcolor='white')

plot_poly(unique_hat_poly, shape = 'exterior', color = 'orange', name = "Hat")
try:
    plot_poly(unique_hat_poly, shape = 'interior', color = 'white', name = None)
except:
    print("No interior polygons")
plot_poly(country_poly, color = 'black', name = country)

fig.update_xaxes(range = [-2.5,2.5])
fig.update_yaxes(range = [-1.25,1.25])

plot(fig)
