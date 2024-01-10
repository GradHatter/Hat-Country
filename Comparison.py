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

#-------------------------------------------------------------------------------------
#normalize and center the hat

#find area of the hat
hat_raw_poly = Polygon(list(zip(hat['x'],hat['y'])))
hat_area = hat_raw_poly.area

#normalize hat area to 1
hat['x_norm'] = hat['x']*np.sqrt(1/(hat_area))
hat['y_norm'] = hat['y']*np.sqrt(1/(hat_area))

#find the center of the normalized hat
hat_norm_poly = Polygon(list(zip(hat['x_norm'],hat['y_norm'])))
hat_center = np.array(hat_norm_poly.centroid.coords)[0]

#center the hat
hat['x_cent'] = hat['x_norm'] - hat_center[0]
hat['y_cent'] = hat['y_norm'] - hat_center[1]

#make the centered hat coordinates into a polygon
hat_polygon = Polygon(list(zip(hat['x_cent'],hat['y_cent'])))

#--------------------------------------------------------------------------------
def norm_cent_poly(poly, country = None):
    c_area = poly.area #find area of the polygon

    #makes 2 rectangles on the + and - boundries
    #goes from noth pole but stops at -75deg long to exclude antartica
    side1 = Polygon([(180,90), (180,-75), (179,-75), (179,90)])
    side2 = Polygon([(-180,90), (-180,-75), (-179,-75), (-179,90)])

    #check if polygon wraps around the international dateline
    if poly.intersects(side1) and poly.intersects(side2):
        edge_overlap = True
    else:
        edge_overlap = False

    blobs = []
    if poly.geom_type == 'MultiPolygon':
        for i in range(len(poly.geoms)):
            #splits each subpolygon into x and y coordinates
            x,y = poly.geoms[i].exterior.xy
            x_corr = np.array(x)
            y_corr = np.array(y)
            #fixes countries arbitrarily split by the international dateline
            if edge_overlap is True:
                x_corr = np.where(x_corr < 0, x_corr+360, x_corr)
            
            #normalize country area to 1
            cx_norm = x_corr*np.sqrt(1/(c_area))
            cy_norm = y_corr*np.sqrt(1/(c_area))

            blobs.append(Polygon(list(zip(cx_norm, cy_norm))))

        #Find the center of the normalized country
        country_norm = MultiPolygon(blobs)
        cent = np.array(country_norm.centroid.coords)[0]

        blobs = []
        for i in range(len(country_norm.geoms)):
            #splits each subpolygon into x and y coordinates
            x,y = country_norm.geoms[i].exterior.xy
            x = np.array(x)
            y = np.array(y)

            #centers normalized country shape
            cx_cent = x - cent[0]
            cy_cent = y - cent[1]

            blobs.append(Polygon(list(zip(cx_cent, cy_cent))))

        #makes the centered and normalized country into one multipolygon
        country_blob = MultiPolygon(blobs)

    elif poly.geom_type == 'Polygon':
        #splits polygon into x and y coordinates
        x,y = poly.exterior.xy
        x_corr = np.array(x)
        y_corr = np.array(y)

        #fixes countries that cross the international dateline
        if edge_overlap is True:
                x_corr = np.where(x_corr < 0, x_corr+360, x_corr)
            
        #normalizes country area to 1
        cx_norm = x_corr*np.sqrt(1/(c_area))
        cy_norm = y_corr*np.sqrt(1/(c_area))

        #find normalized country's center
        country_norm = Polygon(list(zip(cx_norm, cy_norm)))
        cent = np.array(country_norm.centroid.coords)[0]

        #splits normalized polygon into x and y coordinates
        x,y = country_norm.exterior.xy
        x = np.array(x)
        y = np.array(y)

        #center polygon to 0,0

        cx_cent = x - cent[0]
        cy_cent = y - cent[1]

        country_blob = Polygon(list(zip(cx_cent, cy_cent)))

    return country_blob

#-------------------------------------------------------------------------

def find_similarity_score(hat_polygon, country_poly, show = False):

    unique_hat_poly = hat_polygon.difference(country_poly)

    #total hat area-remaining hat area = % overlap
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
            #plot the holes (ideally in the same color as the background)
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
        #plot the holes (ideally in the same color as the background)    
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

#initialize columns of all zeros
world['Similarity Score'] = 0
world['Subtracted Hat Poly'] = 0
world['Norm Cent Country'] = 0
world['Rotation'] = 0
world['Flip'] = 0

similarity_dict = {}
country_list = world['name'].tolist()

for country in country_list:
    #get country geometry then normalize and center
    c = world[(world.name == country)]['geometry'].item()
    c = wkt.loads(c)
    country_poly = norm_cent_poly(c, country == country)

    #rotate the hat every 5 degrees and check overlap
    #keep track of conditions resulting in the highest overlap percentage
    best = [0,0,0,0] #[unique_hat_poly, similarity_score, i*10, flip]
    #rotate the hat every 5 degrees and check overlap
    for i in range(0,360,5):
        hat_polygon = affinity.rotate(hat_polygon, i)
        #flip the hat along the y axis and check again for overlap
        for flip in [True,False]:
            if flip is True:
                hat_polygon = transform(lambda x, y, z=None: (x, -y), hat_polygon)

                #get non-overlapping hat polygon and similarity percentage as decimal
                unique_hat_poly, similarity_score = find_similarity_score(hat_polygon,
                                                                        country_poly,
                                                                        show = False)
                
                #if current conditions are better than the previous best, update best
                if similarity_score > best[1]:
                    best = [unique_hat_poly, similarity_score, i, flip]
                else:
                    pass

            elif flip is False:
                #get non-overlapping hat polygon and similarity percentage as decimal
                unique_hat_poly, similarity_score = find_similarity_score(hat_polygon,
                                                                        country_poly,
                                                                        show = False)
                
                #if current conditions are better than the previous best, update best
                if similarity_score > best[1]:
                    best = [unique_hat_poly, similarity_score, i, flip]
                else:
                    pass

    #update world df with best conditions
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

max_overlap = world.loc[world['Similarity Score'].idxmax()]
min_overlap = world.loc[world['Similarity Score'].idxmin()]

#display the countries with the most and least overlap
print(f"{max_overlap['name']} is the most hat-like country with a similarity of {max_overlap['Similarity Score']:,.2f}%")
print(f"{min_overlap['name']} is the least hat-like country with a similarity of {min_overlap['Similarity Score']:,.2f}%")


country = max_overlap['name']
#unique_hat_poly = wkt.loads(world[(world.name == country)]['Subtracted Hat Poly'].item())
unique_hat_poly = world[(world.name == country)]['Subtracted Hat Poly'].item()
country_poly = world[(world.name == country)]['Norm Cent Country'].item()

world.to_csv("world_similarity.csv")

#-----------------------------------------------------------------------------
#plot the country and the non-overlapping portions of the hat
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
