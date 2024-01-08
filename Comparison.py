import plotly.express as px
import pandas as pd
import numpy as np
from numpy import pi,sin, cos, sqrt, arctan2

from plotly.offline import plot
import plotly.graph_objects as go

from shapely import MultiPolygon, Polygon, affinity, wkt,wkb
import shapely.geometry as sg
import shapely.ops as so


import geopandas as gpd

import matplotlib.pyplot as plt

filename = "file.txt"
#read in file of all countries' data
world = pd.read_csv(filename)

#read in hat outline csv
hat = pd.read_csv("T:\Hat_Outline\Hat_outline.csv")

#select a country
#country = "United States of America"
country = "Russia"

#get data points as a shape
shape = world[(world.name == country)]['geometry'].to_list()[0]

polygonarray = []

# Skip either first 10 or 16 characters if polygon or multipolygon
if shape[0] == 'M':
    i = 16
    xstart = 16
    xend = 16
    ystart = 16
    print('Multipolygon')

# If geometry is Polygon
elif shape[0] == 'P':
    i = 10
    xstart = 10
    xend = 10
    ystart = 10
    print('Polygon')

#Print whatever the shape variable begins with
else:
    print('Error: Shape begins with', shape[0][0],type(shape[0]))

# If geometry is Multipolygon, make list of tuples with "None" seperating shapes 
while i < len(shape):
        if shape[i] == ')':
            if shape[i+2] == ',':
                yend = i
                x = float(shape[xstart:xend])
                y = float(shape[ystart:yend])
                polygonarray.append((x,y))
                xstart = i+6
                polygonarray.append((None, None))
                i+=6
                continue
            elif shape[i+2] ==')':
                yend = i
                x = float(shape[xstart:xend])
                y = float(shape[ystart:yend])
                polygonarray.append((x,y))
                break
            else:
                print('Error: shape[i+2] = {}'.format(shape[i+2]))
                break
        elif shape[i] == ' ':
            xend = i
            ystart = i+1
            i += 1
        elif shape[i] == ',':
            yend = i
            x = float(shape[xstart:xend])
            y = float(shape[ystart:yend])
            polygonarray.append((x,y))
            xstart = i+2
            i += 1
        else:
            i += 1
 
df = pd.DataFrame(data = polygonarray, columns = ['x','y']) 

#fix Russia's data order
if country == 'Russia':
    i = 0
    while i<len(df):
        if df['x'][i] < 0:
            df['x'][i] = df['x'][i] + 360
            pass
        else:
            pass
        i+=1

    df = pd.concat([
                    df.iloc[613:618,:],
                    df.iloc[611:613,:],
                    df.iloc[3:5,:],
                    df.iloc[0:3,:],
                    df.iloc[5:6,:],
                    df.iloc[6:266,:],
                    df.iloc[607:611,:],
                    df.iloc[586:607,:],
                    df.iloc[266:586,:],], ignore_index = True)
    #print(min(df['x']), max(df['x']))

#print where the breaks between each shape are
#print(df[df['x'].isnull()])

#find center of the shape
def center_shape(df):
    centriod = (df['x'].mean(), df['y'].mean())

    df['x_cent'] = df['x']-centriod[0]
    df['y_cent'] = df['y']-centriod[1]

    x,y = df['x_cent'].values, df['y_cent'].values

    df['r'] = np.sqrt(x**2+y**2)
    df['theta'] = np.arctan2(y,x)
    df['theta'] = df['theta']*180/np.pi


    df['360theta'] = np.abs(df['theta'])
    '''
    i = 0
    while i < len(df):
        if df['theta'].iloc[i] < 0:
            df['360theta'].iloc[i] = 180 - df['theta'].iloc[i]
        else:
            df['360theta'].iloc[i] = df['theta']
        i+=1
     '''  
    return df
#--------------------------------------------------------------------------------
#find area using shoelace method
def PolyArea(x,y):
    x= np.array(x)
    y = np.array(y)
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

#---------------------------------------------------------------------------------
#find country area by adding each section's area
def multipoly_area(x,y):
    #find where the breaks between shapes are
    listna = x[x.isnull()].index.to_list()
    c_area = 0
    for i in range(len(listna)):
        if i==len(listna)-1:
            c_area += PolyArea(x[listna[i-1]+1:listna[i]], y[listna[i-1]+1:listna[i]])
            #c_area += PolyArea(x[listna[i]+1:], y[listna[i]+1:])
        elif i==0:
            c_area += PolyArea(x[:listna[i]], y[:listna[i]])
        else:
            c_area += PolyArea(x[listna[i-1]+1:listna[i]], y[listna[i-1]+1:listna[i]])

    return c_area
#------------------------------------------------------------------------------------

#find area of the hat
hat_area = PolyArea(hat['x'],hat['y'])

#find the area of the country
c_area = multipoly_area(df['x'],df['y'])

#normalize hat area to 1
hat['x_norm'] = hat['x']*np.sqrt(1/(hat_area))
hat['y_norm'] = hat['y']*np.sqrt(1/(hat_area))

#Check it is normalized
#print(PolyArea(hat['x_norm'],hat['y_norm']))

#normalize country area to 1
df['x_norm'] = df['x']*np.sqrt(1/(c_area))
df['y_norm'] = df['y']*np.sqrt(1/(c_area))

#check if normalized
#print(multipoly_area(df['x_norm'],df['y_norm']))

hat['x_cent'] = hat['x_norm'] - hat['x_norm'].mean()
hat['y_cent'] = hat['y_norm'] - hat['y_norm'].mean()

df['x_cent'] = df['x_norm'] - df['x_norm'].mean()
df['y_cent'] = df['y_norm'] - df['y_norm'].mean()

fig = go.Figure([
    go.Scatter(x = df['x_cent'], y = df['y_cent'], 
                           fill = 'toself'),
    go.Scatter(x = hat['y_cent'], y = hat['x_cent'],
                           fill = 'toself')
                           ])

fig.update_xaxes(range = [-2,2])
fig.update_yaxes(range = [-.75,.75])

#fig = px.line_polar(df[60:258], r='r', theta="theta", line_close=True,
#                    color_discrete_sequence=px.colors.sequential.Plasma_r,
#                    template="plotly_dark",)

#plot(fig)

#----------------------------------------------------------------------------

#make list of tuples from dataframe
column1 = 'x_cent'
column2 = 'y_cent'
hat_list = list(zip(hat[column1], hat[column2]))
country_poly_list = []

listna = df[df[column1].isnull()].index.to_list()
for i in range(len(listna)):
        if i==0:
            country_poly_list.append(Polygon(df[[column1,column2]][:listna[i]]))
        else:
            country_poly_list.append(Polygon(df[[column1,column2]][listna[i-1]+1:listna[i]]))


hat_polygon = Polygon(hat[[column1,column2]])

hat_polygon = affinity.rotate(hat_polygon, 90) #rotate polygon


#hat_polygon = hat_polygon.difference(country_poly_list[5])
#hat_polygon = hat_polygon.difference(country_poly_list[3])
#hat_polygon = hat_polygon.difference(country_poly_list[-1])
hat_polygon = Polygon(hat_polygon.exterior.coords, [country_poly_list[-10].exterior.coords])
#hat_polygon = hat_polygon.difference(country_poly_list[1])
print(hat_polygon.area)
#hat_polygon.plot()
#plt.show()
'''
fig = go.Figure()

#Check if the object is a polygon or Multipolygon
if hat_polygon.geom_type == 'MultiPolygon':
    print('Multipolygon')
    for geom in list(hat_polygon.geoms):
        ys, xs = geom.exterior.xy

        fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys), 
                                fill = 'toself'))
elif hat_polygon.geom_type == 'Polygon':
    print('Polygon')
    ys, xs = hat_polygon.exterior.xy

    fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys), 
                                fill = 'toself'))
else:
    raise IOError('Shape is not a polygon.')
'''
'''
for geom in list(hat_polygon.geoms):
    ys, xs = geom.exterior.xy

    fig.add_trace(go.Scatter(x = np.array(xs), y = np.array(ys), 
                            fill = 'toself'))
'''
'''
#fig.add_trace(go.Scatter(x = hat['y_cent'], y = hat['x_cent'],
#                            fill = 'toself'))
fig.add_trace(go.Scatter(x = df['y_cent'], y = df['x_cent'],
                            fill = 'toself'))

y1s, x1s = country_poly_list[-10].exterior.xy

fig.add_trace(go.Scatter(x = np.array(x1s), y = np.array(y1s), 
                                fill = 'toself'))
plot(fig)
'''

c = world['geometry'].apply(wkt.loads)#.values[0]
c = world[(world.name == country)]['geometry'].item()
c = wkt.loads(c)

blobs = []

for i in range(len(c.geoms)):
    x,y = c.geoms[i].exterior.xy
    x = np.array(x)
    x_corr = np.where(x < 0, x+360, x)
    y_corr = np.array(y)
    blobs.append(Polygon(list(zip(x_corr, y_corr))))

country_blob = MultiPolygon(blobs)

fig, axs = plt.subplots()

for geom in country_blob.geoms:    
    x2s, y2s = geom.exterior.xy    
    axs.fill(x2s, y2s)

plt.show()
