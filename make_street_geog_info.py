# %%
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from shapely.geometry import Point, LineString

from scipy.interpolate import Rbf



# %%
# jc_ruas = gpd.read_file('/home/georgynio/Documentos/shapes_diversos/bairros/ruas_max_20m-JC.shp')
jc_ruas = gpd.read_file('/home/georgynio/Documentos/disco_2/shapes_diversos/vias_munich/jc_polyline.shp')
jc_building = gpd.read_file('/home/georgynio/Documentos/disco_2/shapes_diversos/vias_munich/building_JC.shp')


# %%
# jc_poly = gpd.read_file('sirane/dados_simulacao/Jardim_Camburi/GEOMETRIA/Shapefiles/Shapefiles/jc_polyline.shp')
# the column names is founded in SIRANE description, at the bellow link
# http://air.ec-lyon.fr/SIRANE/Article.php?Id=SIRANE_File_ReseauShpDbf&Lang=EN
jc_ruas = jc_ruas.to_crs(4326)
jc_building = jc_building.to_crs(4326)

jc_building['height'] = jc_building.height.fillna(0).astype('int')
jc_building.head()

# %%
pd.DataFrame(jc_ruas.Nome.unique()).count()

# %%
coordenadas = jc_building.geometry.apply(lambda p: (p.centroid.y, p.centroid.x)).tolist()
# coordenadas


# %%
jc_building.building_l.fillna(0).astype('int').value_counts(sort=True)*100/len(jc_building.building_l)

# para a construção da caixa vamos a considerar o valor mais alto entre o HG e HD, o resultado é salvo em HM
# segue a mesma ideia, a media entre WG e WD, o valor é salvo em WM

jc_ruas['HM'] = jc_ruas[['HG', 'HD']].max(axis=1)
jc_ruas['W_tot'] = jc_ruas[['WG', 'WD']].sum(axis=1)

# junção das ruas e predios de acordo com a distancia
ruas_buildings = gpd.sjoin_nearest(jc_ruas, jc_building, how='left', distance_col='distancia')

ruas_buildings = ruas_buildings.drop(
    columns=['MODUL_EMIS', 'addr_subur', 'amenity', 'addr_hou_1', 'source', 'index_right',
             'name', 'type', 'WG', 'WD',  'NDDEB', 'NDFIN', 'building', 'addr_stree'])

ruas_buildings = ruas_buildings.rename(columns={'ID_left': 'ID'})


ruas_buildings
# ruas_buildings['altura'] = ruas_buildings['']

# Split the linestrings at the intersections
split_lines = jc_ruas.geometry.apply(lambda x: x.intersection(jc_ruas.unary_union))

# Flatten the resulting list of linestrings into a single GeoDataFrame
split_lines_gdf = gpd.GeoDataFrame({'geometry': split_lines.explode()})

# Create a dictionary to store the segments that are connected to each intersection
segments = {}

# Iterate through the segments and add them to the dictionary
for index, row in split_lines_gdf.iterrows():
    start_point = tuple(row['geometry'].coords[0])
    end_point = tuple(row['geometry'].coords[-1])
    if start_point in segments:
        segments[start_point].append(index[0])
    else:
        segments[start_point] = [index[0]]
    if end_point in segments:
        segments[end_point].append(index[0])
    else:
        segments[end_point] = [index[0]]

# %%
geometry = [Point(p) for p in segments.keys()]
intersection_jc = gpd.GeoDataFrame(geometry=geometry, crs='EPSG:4326')
intersection_jc['streets'] = segments.values()
for i, row in intersection_jc.iterrows():
    intersection_jc.loc[i, 'id'] = i+1
    intersection_jc.loc[i, 'num_ruas'] = len(row['streets'])
    intersection_jc.loc[i, 'long'] = row.geometry.x
    intersection_jc.loc[i, 'lat'] = row.geometry.y


# %%
linestring_nodes = {}

# Loop through each linestring
for i, row in jc_ruas.iterrows():
    # Find the intersection that is the initial node for the linestring
    jc_ruas.at[i, 'ini_node'] = intersection_jc[intersection_jc.intersects(Point(row["geometry"].coords[0]))]["id"].iloc[0]
    # Find the intersection that is the final node for the linestring
    jc_ruas.at[i, 'end_node'] = intersection_jc[intersection_jc.intersects(Point(row["geometry"].coords[-1]))]["id"].iloc[0]
    # Store the initial and final nodes in the dictionary

jc_ruas.head()

# %%
nearest = gpd.sjoin_nearest(jc_ruas, jc_building)
nearest.index = nearest['ID_left']
nearest = nearest.sort_index()
height = list(nearest.iloc[jc_ruas.ID-1]['height'])


# %%
ruas_buildings

# %%
# colocando altura 3 metros caso a não existe predio do lado
ruas_buildings.loc[ruas_buildings['height']==0, 'height'] = 3

# Escenario específico
escenario = 'cenario-wudapt'

# Definir las alturas correspondientes a cada escenario
alturas_por_escenario = {
    'cenario-base': None,
    'cenario-1': 6,
    'cenario-2': 9,
    'cenario-3': 12,
    'cenario-4': 15,
    'cenario-5': 18,
    'cenario-wudapt': 10,
}

# Colocar alturas según el escenario
altura_predeterminada = alturas_por_escenario.get(escenario, 3)
if 'wudapt' in escenario:
    ruas_buildings.loc[:, 'height'] = altura_predeterminada
else:
    ruas_buildings.loc[ruas_buildings['height'] < altura_predeterminada, 'height'] = altura_predeterminada

# prepare the street-geog-info.dat file
# second case
# we use 3 meters as unique height or 0.1 meters to represent opened area
cabeca1 = '#id,length,width,height'
df1 = pd.DataFrame(columns=cabeca1.split(','))

df1['#id'] = ruas_buildings['ID']
df1['length'] = ruas_buildings['LENGTH']
df1['width'] = ruas_buildings['W_tot']
df1['height'] = ruas_buildings['height']
nome_arquivo = f'street-geog-{escenario}.dat'
df1.to_csv(nome_arquivo, sep='\t', index=False)



