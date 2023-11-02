# Imports from standard libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import altair as alt
import seaborn as sns
from scipy.stats import pearsonr
import requests
import json
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from API_puller import API_puller
from datetime import datetime, date, time
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import mpld3
import streamlit.components.v1 as components
import random
import plotly.graph_objects as go

header = st.container()
dataset =st.container()
st.title('Energy Disaggregation using BAS Trend Data')

# col1, col2, col3 = st.columns(3)

# with col1:
# st.header("A cat")   
#matplotlib inline
header = st.container()
dataset =st.container()
# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()

# @st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# building_number = st.multiselect('Insert building number',[3642,3035])
building_number = 3642 # Building
st.write('The building number is ', building_number)


search_term="HeatingValve"
# search_term = st.selectbox(
#     'Trend data?',
#     ('HeatingValve', 'CoolingValve'))

st.write('You selected:', search_term)

st.header("Date Range")
start_date = st.date_input('start date')
end_date = st.date_input('end date')
print(start_date)
print(end_date)
# search_term = 'HeatingValve' # Keyword search term
# Import API puller from supplementary file

################################ Supplementary functions#####################
def check_response(r):
    '''Checks to ensure the expected response is received

    The accepted response from the API from the API is response [200] this
    function outputs raises an error if any other response is received.
    '''
    if r.status_code == 200:
        return None
    else:
        raise ImportError('Received: [{}], Expected: [<Response [200]>]'.format(r))
#########################################################################
def check_response(r):
    '''Checks to ensure the expected response is received

    The accepted response from the API from the API is response [200] this
    function outputs raises an error if any other response is received.
    '''
    if r.status_code == 200:
        return None
    else:
        raise ImportError('Received: [{}], Expected: [<Response [200]>]'.format(r))
############################################################################## 
def print_n_lines(json_var, n_lines=20, indent=2):
    '''Pretty prints n lines of json file.

    This is used to make the outputs more compact
    '''
    pretty_str = str(json.dumps(json_var, indent=indent))
    length = len(pretty_str.splitlines())

    print('First {} / {} lines of json file:\n'.format(n_lines, length))
    for line in pretty_str.splitlines()[:n_lines]:
        print(line)
    print('..............')       
# Load text file
with open('login_info_v1.txt') as f:
    login_info = json.loads(f.read())

# Assign variables
api_key = login_info['api_key']
client_id = login_info['api_key'] # The client ID is the same as the API key
client_secret = login_info['client_secret']
print('Login info successfully downloaded')

######################################################################
# Request access token using client_id, and client_secret
url = 'https://login-ca-central-prod.coppertreeanalytics.com/oauth2/token'

my_header = {'content-type': 'application/x-www-form-urlencoded'}
my_data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
    'audience': 'organize'
}
r = requests.post(url, headers=my_header, data=my_data)
check_response(r)
access_token = r.json()['access_token']

# Save in jWt header fomrat
jwt_header = {'Authorization': 'Bearer ' + access_token}
print('Access token has been obtained')


#######################################################################
# Example inputs
# Jean Talon id= 3642


#######################################################################

# Initial API query gets sensor count
url = 'https://kaizen.coppertreeanalytics.com/yana/mongo/objects/?' \
        'building={}&object_type=TL&min_device_index=1&page_size=1'.format(building_number)
r = requests.get(url, headers=jwt_header)
count = r.json()['meta']['pagination']['count']

# Second API query gets full sensor list
url = 'https://kaizen.coppertreeanalytics.com/yana/mongo/objects/?' \
        'building={}&object_type=TL&min_device_index=1&page_size={}'.format(building_number, count)
r = requests.get(url, headers=jwt_header)

# Convert to pandas dataframe
df = pd.DataFrame.from_dict(r.json()['results'])[['_id', 'Object_Name']]

print(df)
#########################################################################]
# Filter based on keyword
df_filtered = df[df['Object_Name'].str.contains(search_term)].reset_index(drop=True)

########################### Inputs ######################
# The download using the batch API puller
trend_log_data = API_puller(
    trend_log_list=df_filtered,
    API_key=api_key,
    # date_range=['2019-01-01', '2020-01-01'],
    date_range=[start_date,end_date],
    resample=60
    )
##########################################################################
# Plot data retrieved from the API
print(trend_log_data)
fig, ax = plt.subplots(figsize=(15,8))
# fig = plt.figure(figsize=(15,8))
st.line_chart(trend_log_data)
selected_column = st.selectbox('Select a column to plot:', trend_log_data.columns)
st.line_chart(data=trend_log_data, y=selected_column)
# plt.plot(trend_log_data)
# plt.xlim(trend_log_data.index[0],trend_log_data.index[1000])
# plt.title('Data downloaded from API')
# plt.ylabel('Heating Valve %')
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fancybox=True, shadow=True);
# plt.show()s
# st.pyplot(fig)
    #########################################################################
# with col2:
st.header('Heating Energy Use (MJ)')
search_term2='Jean Talon- Steam'
df_energy = df[df['Object_Name'].str.contains(search_term2)].reset_index(drop=True)
trend_log_data_energy = API_puller(
    trend_log_list=df_energy,
    API_key=api_key,
    # date_range=['2019-01-01', '2020-01-01'],
    date_range=[start_date,end_date],
    resample=60
)

print(trend_log_data_energy)
st.line_chart(trend_log_data_energy)
#########################################

# search_term3='OutdoorTemp'
# df_otemp = df[df['Object_Name'].str.contains(search_term3)].reset_index(drop=True)
# trend_log_data_temp = API_puller(
#     trend_log_list=df_otemp,
#     API_key=api_key,
#     # date_range=['2019-01-01', '2020-01-01'],
#     date_range=[start_date,end_date],
#     resample=60
# )
# print(trend_log_data_temp)
# st.line_chart(trend_log_data_temp)

############################ Weather Data #######################

st.header('Outdoor Air Temperature (degC)')
# Available time formats: LST/UTC
timeformat = 'LST'

# check https://power.larc.nasa.gov/#resources for full list of parameters and modify as needed

# ALLSKY_SFC_SW_DNI - direct normal irradiance (W/m2)
# ALLSKY_SFC_SW_DIFF - diffuse horizontal irradiance (W/m2)
# T2M - temperature 2 m above ground (degC)
# RH2M - relative humidity 2 m above ground level (m/s)
# WS2M - wind speed 2 m above ground level (m/s)

# params = 'ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,RH2M,WS2M'
params = 'T2M'
#Always use RE (renewable energy) for this purpose
community = 'RE' 
#Obtain LAT/LON from google maps
location = {
    'latitude':'45.73906',
    'longitude':'-75.73906'
    }
# Start/end time in format: 'YYYYMMDD'


# sTime = '20180101'
# eTime = '20190101'
sTime = str(start_date)
sTime = sTime.replace('-','')
eTime = str(end_date)
eTime = eTime.replace('-','')
print(eTime)

#%% API call for given lat/long
cwd = Path.cwd()
path = cwd.__str__()
url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=JSON'
data = requests.get(url)

data = data.json()
data = pd.DataFrame((data['properties']['parameter']))
data = data.set_index(pd.to_datetime(data.index, format='%Y%m%d%H'))

st.line_chart(data)
print(data)
##############################Regression Model###########

st.header('Heating Disaggregation Results')

df_model = pd.merge(trend_log_data, data['T2M'], left_index=True, right_index=True)
df_model = pd.merge(df_model,trend_log_data_energy, left_index=True, right_index=True)
predictors = df_model.drop(['Jean Talon- Steam'], axis=1)
response = df_model['Jean Talon- Steam']
print(df_model)
print(predictors.shape[1])
print(predictors)
# print(response)
# print(data['T2M'])
def rmse_HtgMdl(x): #Heating disagg model
    h = 0
    for i in range(predictors.shape[1]-1):
        h = x[i]*predictors.iloc[:,i] + h

    h = h + x[i+1]*(predictors.iloc[:,i+1]-x[i+2])*(predictors.iloc[:,i+1]-x[i+2]) + x[i+3]
    return np.sqrt(((response - h) ** 2).mean())

x0 = np.zeros(predictors.shape[1]+2)
b = (0.0,10)
bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
solution_Htg = minimize(rmse_HtgMdl,x0,method='SLSQP',bounds=bnds)
print(solution_Htg)
print(predictors.columns)
heating_ahu = pd.DataFrame()

# for i in range(0,predictors.shape[1]):
#     heating_ahu = pd.concat([heating_ahu,(predictors.iloc[:,i]*solution_Htg[i])], axis=1)

# heating_perimeter = solution_Htg[-2] * (predictors.iloc[:,-1]-solution_Htg[-2])*(predictors.iloc[:,-1]-solution_Htg[-2]) + solution_Htg[-1] *np.ones(df_model['AH7_HeatingValve'].shape)
# labels = trend_log_data.columns
# fig, axs = plt.subplots(14,sharex=True, figsize=(15,30))
# for i in range(0,14):
#     fig.suptitle('Heating Disaggregation ')
#     axs[i].plot(solution_Htg.x[i]*predictors.iloc[:,i], color='red',linewidth=1, linestyle='dashed')
#     axs[i].set(ylabel='kW')
#     axs[i].set_title(labels[i])
#   # axs[i].set_xlim(2000,2400)
#     fig.tight_layout(pad=2)
#     fig.subplots_adjust(top=0.9)
# st.plotly_chart(fig)

df_Htg_ahu = pd.DataFrame(columns = trend_log_data.columns)

for i in range(predictors.shape[1]-1):
    this_column = df_Htg_ahu.columns[i]
    df_Htg_ahu[this_column] = solution_Htg.x[i]*predictors.iloc[:,i]

print(df_Htg_ahu)

heating_disagg = pd.DataFrame()
heating_perimeter_other = solution_Htg.x[-3]*(predictors.iloc[:,-1]-solution_Htg.x[-2])*(predictors.iloc[:,-1]-solution_Htg.x[-2]) + solution_Htg.x[-1]
print(heating_perimeter_other)

heating_disagg = pd.DataFrame()
heating_disagg = pd.merge(df_Htg_ahu,heating_perimeter_other, left_index=True, right_index=True)
heating_disagg['Perimeter Heaters and Others'] = heating_disagg['T2M']
heating_disagg = heating_disagg.drop(['T2M'], axis=1)
BLDGE_AREA =70,970
print(heating_disagg)

kpi_htg_ahu = df_Htg_ahu.sum()
kpi_htg_perimeter_other = heating_perimeter_other.sum()
print(kpi_htg_ahu)
print(kpi_htg_perimeter_other)

source_htg = pd.DataFrame()
source_htg.index = kpi_htg_ahu.index
source_htg['AHU Heating Energy(MJ)'] =kpi_htg_ahu.values
st.bar_chart(source_htg)
# st.bar_chart(kpi_htg_ahu)
# label = ['Heating Energy','AH7_HeatingValve', 'AH8_HeatingValve', 'AH9_HeatingValve',
#          'AH10_HeatingValve', 'AH11_HeatingValve', 'AH12_HeatingValve',
#          'AH13_HeatingValve', 'AH1_HeatingValve', 'AH2_HeatingValve',
#          'AH14_HeatingValve', 'AH15_HeatingValve', 'AH16_HeatingValve',
#          'AH4_HeatingValve', 'AH5_HeatingValve']
# source = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# value = kpi_htg_ahu.values
# print(label)
# # data to dict, dict to sankey
# link = dict(source = source, target = target, value = value)
# node = dict(label = label, pad=50, thickness=7)
# data = go.Sankey(link = link, node=node)
# # plot
# fig = go.Figure(data)
# fig
############Cooling#############################
# with col3:
st.header('Cooling Energy (MJ)')
search_term3='Jean Talon- Chilled Water'
df_cooling = df[df['Object_Name'].str.contains(search_term3)].reset_index(drop=True)
trend_log_data_cooling = API_puller(
    trend_log_list=df_cooling,
    API_key=api_key,
    # date_range=['2019-01-01', '2020-01-01'],
    date_range=[start_date,end_date],
    resample=60
)

print(trend_log_data_cooling)
st.line_chart(trend_log_data_cooling)
############Cooling valve position #############################
st.header('Cooling Valve Position (%)')
search_term4='CoolingValve'
df_coolingvalve = df[df['Object_Name'].str.contains(search_term4)].reset_index(drop=True)
trend_log_data_coolingvalve = API_puller(
    trend_log_list=df_coolingvalve,
    API_key=api_key,
    # date_range=['2019-01-01', '2020-01-01'],
    date_range=[start_date,end_date],
    resample=60
)

print(trend_log_data_coolingvalve)
st.line_chart(trend_log_data_coolingvalve)
selected_column = st.selectbox('Select a column to plot:', trend_log_data_coolingvalve.columns)
st.line_chart(data=trend_log_data_coolingvalve, y=selected_column)
############Cooling regression Model###########################
st.header('Cooling Disaggregation Results')

df_model_cooling = pd.merge(trend_log_data_coolingvalve, data['T2M'], left_index=True, right_index=True)
df_model_cooling = pd.merge(df_model_cooling,trend_log_data_cooling, left_index=True, right_index=True)
predictors = df_model_cooling.drop(['Jean Talon- Chilled Water'], axis=1)
response = df_model_cooling['Jean Talon- Chilled Water']
print(df_model_cooling)
print(predictors.shape[1])
print(predictors)
# print(response)
# print(data['T2M'])
def rmse_ClgMdl(x): #Heating disagg model
    h = 0
    for i in range(predictors.shape[1]-1):
        h = x[i]*predictors.iloc[:,i] + h

    h = h + x[i+1]*(predictors.iloc[:,i+1]-x[i+2])*(predictors.iloc[:,i+1]-x[i+2]) + x[i+3]
    return np.sqrt(((response - h) ** 2).mean())

x0 = np.zeros(predictors.shape[1]+2)
b = (0.0,10)
bnds = (b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b)
solution_Clg = minimize(rmse_ClgMdl,x0,method='SLSQP',bounds=bnds)
print(solution_Clg)
print(predictors.columns)
cooling_ahu = pd.DataFrame()

# for i in range(0,predictors.shape[1]):
#     heating_ahu = pd.concat([heating_ahu,(predictors.iloc[:,i]*solution_Htg[i])], axis=1)

# heating_perimeter = solution_Htg[-2] * (predictors.iloc[:,-1]-solution_Htg[-2])*(predictors.iloc[:,-1]-solution_Htg[-2]) + solution_Htg[-1] *np.ones(df_model['AH7_HeatingValve'].shape)
# labels = trend_log_data.columns
# fig, axs = plt.subplots(14,sharex=True, figsize=(15,30))
# for i in range(0,14):
#     fig.suptitle('Heating Disaggregation ')
#     axs[i].plot(solution_Htg.x[i]*predictors.iloc[:,i], color='red',linewidth=1, linestyle='dashed')
#     axs[i].set(ylabel='kW')
#     axs[i].set_title(labels[i])
#   # axs[i].set_xlim(2000,2400)
#     fig.tight_layout(pad=2)
#     fig.subplots_adjust(top=0.9)
# st.plotly_chart(fig)

df_Clg_ahu = pd.DataFrame(columns = trend_log_data_coolingvalve.columns)

for i in range(predictors.shape[1]-1):
    this_column = df_Clg_ahu.columns[i]
    df_Clg_ahu[this_column] = solution_Clg.x[i]*predictors.iloc[:,i]

print(df_Clg_ahu)

cooling_disagg = pd.DataFrame()
cooling_perimeter_other = solution_Clg.x[-3]*(predictors.iloc[:,-1]-solution_Clg.x[-2])*(predictors.iloc[:,-1]-solution_Clg.x[-2]) + solution_Clg.x[-1]
print(cooling_perimeter_other)

cooling_disagg = pd.DataFrame()
cooling_disagg = pd.merge(df_Clg_ahu,cooling_perimeter_other, left_index=True, right_index=True)
cooling_disagg['Perimeter Heaters and Others'] = cooling_disagg['T2M']
cooling_disagg = cooling_disagg.drop(['T2M'], axis=1)
BLDGE_AREA =70,970
print(cooling_disagg)

kpi_clg_ahu = df_Clg_ahu.sum()
kpi_clg_perimeter_other = cooling_perimeter_other.sum()
print(kpi_clg_ahu)
print(kpi_clg_perimeter_other)

source = pd.DataFrame()
source.index = kpi_clg_ahu.index
source['AHU Cooling Energy(MJ)'] =kpi_clg_ahu.values
st.bar_chart(source)
# st.bar_chart(kpi_clg_ahu)


label = ['Cooling Energy','AH7_CoolingValve', 'AH8_CoolingValve', 'AH9_CoolingValve',
            'AH10_CoolingValve', 'AH11_CoolingValve', 'AH12_CoolingValve','AH13_CoolingValve', 'AH14_CoolingValve', 
            'AH2_CoolingValve','AH15_CoolingValve', 'AH16_CoolingValve',
            'Pent_Elev_CoolingValve','AH4_CoolingValve', 'AH5_CoolingValve',
            'Heating Energy','AH7_HeatingValve', 'AH8_HeatingValve', 'AH9_HeatingValve', 'AH10_HeatingValve', 'AH11_HeatingValve'
            ,'AH12_HeatingValve', 'AH13_HeatingValve', 'AH1_HeatingValve', 'AH2_HeatingValve', 'AH14_HeatingValve', 'AH15_HeatingValve',
            'AH16_HeatingValve', 'AH4_HeatingValve', 'AH5_HeatingValve','Perimeter Heating Devices','Unmonitored Cooling']
source = [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  0]
target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# value1 = np.array(kpi_clg_ahu.values)
# value2 = np.array(kpi_htg_ahu.values)
value = [0.000000e+00, 7.125857e+05,1.396961e+05, 8.346263e+05, 1.324155e+06, 3.682884e+05
            ,8.682047e+05, 0.000000e+00, 0.000000e+00, 1.209310e+06,6.243443e+05, 
            9.725472e-05, 1.348454e+05, 1.942690e-07, 6.837227e-08,3.019299e-09, 1.637352e-07,
            0.000000e+00,7.620935e-05, 8.367065e-09, 5.811317e+03, 1.820391e-07,1.421369e+06,
            6.857399e+05,8378372.945483186,2481835.854260325]
print(kpi_htg_ahu)
print(kpi_clg_ahu)
print(kpi_clg_perimeter_other)
print(kpi_htg_perimeter_other)
# value = np.concatenate(value1, value2)
print(value)
# data to dict, dict to sankey

color_link = ['#000000', '#FFFF00', '#1CE6FF', '#FF34FF', '#FF4A46',
             '#008941', '#006FA6', '#A30059','#FFDBE5', '#7A4900', 
             '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF',
             '#997D87', '#5A0007', '#809693', '#FEFFE6', '#1B4400', 
             '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80', '#61615A',
             '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9',
             '#B903AA', '#D16100', '#DDEFFF', '#000035', '#7B4F4B',                
             '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F',
             '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744',
             '#C0B9B2', '#C2FF99', '#001E09', '#00489C', '#6F0062', 
             '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1',
             '#788D66', '#885578', '#FAD09F', '#FF8A9A', '#D157A0',
             '#BEC459', '#456648', '#0086ED', '#886F4C', '#34362D', 
             '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', 
             '#FF913F', '#938A81', '#575329', '#00FECF', '#B05B6F',
             '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00',
             '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C',
             '#772600', '#D790FF', '#9B9700', '#549E79', '#FFF69F', 
             '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465',
             '#922329', '#5B4534', '#FDE8DC', '#404E55', '#0089A3',
             '#CB7E98', '#A4E804', '#324E72', '#6A3A4C'
             ]
link = dict(source = source, target = target, value = value,color=color_link)
node = dict(label = label, pad=35, thickness=20)
data = go.Sankey(link = link, node=node)
# plot
fig2 = go.Figure(data)
fig2.update_layout(
    hovermode='x',
    font=dict(size=10, color='white'),
    paper_bgcolor='#51504f'
)
fig2