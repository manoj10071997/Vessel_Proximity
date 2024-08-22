#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


# In[4]:


# Load and preprocess the data
data = pd.read_csv('sample_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values(by=['mmsi', 'timestamp']).reset_index(drop=True)


# In[5]:


# Convert lat/lon to radians for BallTree
data['lat_rad'] = np.radians(data['lat'])
data['lon_rad'] = np.radians(data['lon'])


# In[6]:


# Define the Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# In[ ]:


# Distance threshold in kilometers
distance_threshold = 1.0
distance_threshold_rad = distance_threshold / 6371.0


# In[ ]:


# Create a BallTree for efficient spatial search
coords = data[['lat_rad', 'lon_rad']].values
ball_tree = BallTree(coords, metric='haversine')


# In[ ]:


# Cluster vessels by time: Create time windows of 1 hour
data['time_window'] = data['timestamp'].dt.floor('H')
grouped = data.groupby('time_window')


# In[15]:


# Find proximity events within time windows
def find_proximity_events(grouped_data, distance_threshold_rad):
    proximity_events = []
    for _, group in grouped_data:
        group = group.reset_index(drop=True)
        group_coords = group[['lat_rad', 'lon_rad']].values
        group_ball_tree = BallTree(group_coords, metric='haversine')

        for i in range(len(group)):
            indices = group_ball_tree.query_radius([group_coords[i]], r=distance_threshold_rad)[0]
            for j in indices:
                if i != j and group.loc[i, 'mmsi'] != group.loc[j, 'mmsi']:
                    time_diff = abs((group.loc[j, 'timestamp'] - group.loc[i, 'timestamp']).total_seconds())
                    if time_diff <= 3600:
                        distance = haversine(group.loc[i, 'lon'], group.loc[i, 'lat'], group.loc[j, 'lon'], group.loc[j, 'lat'])
                        if distance <= distance_threshold:
                            proximity_events.append({
                                'mmsi': group.loc[i, 'mmsi'],
                                'vessel_proximity': group.loc[j, 'mmsi'],
                                'timestamp': group.loc[i, 'timestamp'],
                                #'timestamp_2': group.loc[j, 'timestamp'],
                                #'distance_km': distance
                            })
    return pd.DataFrame(proximity_events)


# In[16]:


# Execute the algorithm
proximity_df = find_proximity_events(grouped, distance_threshold_rad)
proximity_df


# In[ ]:




