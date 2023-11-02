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

st.title('Energy Disaggregation using BAS Trend Data')

# col1, col2, col3 = st.columns(3)

# with col1:
header = st.container()
dataset =st.container()
# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()

# @st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

st.title("Cooling Data")
# data_Energy = st.file_uploader("Upload Total Energy Data in CSV format", type=["csv"])
#     if data_Energy is not None:
#         st.write(type(data_Energy))
#         file_details = {"filename":data_Energy.name,"filetype":data_Energy.type,"filesize":data_Energy.size}
#         st.write(file_details)
#         df_energy = pd.read_csv(data_Energy,index_col=0)
#         # df_energy.index= pd.to_datetime(df_energy.index)
#         st.write(file_details)
#         st.dataframe(df_energy)
uploaded_file_Clg = st.file_uploader("Upload Cooling Energy Data and BAS in CSV format",type=["csv"])
if uploaded_file_Clg is not None:
    # df_Clg = pd.read_csv('Trend Logs (3).csv',index_col=0)
    df_Clg = pd.read_csv(uploaded_file_Clg,index_col=0)
    df_Clg.index = pd.to_datetime(df_Clg.index)
    df_Clg_2 = df_Clg.fillna(method="Pad")
    df_Clg_3 = df_Clg_2.resample('H').mean()
# plt.plot(df_Clg_2['Jean Talon- Chilled Water']) 
    print(df_Clg_3)
    st.write(df_Clg_3)
    
    # # building_number = st.multiselect('Insert building number',[3642,3035])
    # building_number = 3642 # Building
    # st.write('The building number is ', building_number)

# st.header("Date Range")
# # start_date = st.date_input('start date',min_value=datetime.date(2017, 6, 10), max_value=datetime.date(2019, 1, 1))
# min_date = date(2017, 6, 10)
# max_date = date(2019, 1, 1)
# a_date = st.date_input("Pick a date",  min_value=min_date,max_value=max_date)

# min_value=datetime.date(2021, 1, 1),
    # max_value=datetime.date(2023, 1, 1),
# end_date = st.date_input('end date',min_value=datetime.date(2017, 6, 10), max_value=datetime.date(2019, 1, 1))
# print(start_date)
# print(end_date)
######################

    st.header('Cooling Valves (%)')
    df_valves = df_Clg_3.drop(['Jean Talon- Chilled Water (MJ)'], axis=1)
    # line_chart = alt.Chart(df_valves).mark_line(interpolate='basis').properties(title='Sales of consumer goods')
    # st.altair_chart(line_chart)
    st.line_chart(df_valves)
    selected_column = st.selectbox('Select a column to plot:', df_valves.columns)
    st.line_chart(data=df_valves, y=selected_column)
    # print(df_Htg['Jean Talon- Steam'].iloc[1:1000])
    st.header('Cooling Energy Meter (MJ)')
    st.line_chart(data = df_Clg_3,y='Jean Talon- Chilled Water (MJ)')
    # st.line_chart(data = df_Clg_3, x=df_Clg_3.index , y = "temp_max")
    # df_Clg_3.corr(numeric_only=True)
    # fig = plt.figure() 
    # plt.plot(df_Clg['Jean Talon- Chilled Water']) 

    # st.pyplot(fig)

    ############################ Weather Data #######################
if st.button('Cooling disaggrgation Results'):
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
    

    sTime = '20170609'

    eTime = '20190101'
    # sTime = str(df_Clg.index[0])
    # sTime = sTime.replace('-','')
    # eTime = str(df_Clg.index[-1])
    # eTime = eTime.replace('-','')
    # print(eTime)

    # %% API call for given lat/long
    cwd = Path.cwd()
    path = cwd.__str__()
    url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=JSON'
    data = requests.get(url)

    data = data.json()
    data = pd.DataFrame((data['properties']['parameter']))
    data = data.set_index(pd.to_datetime(data.index, format='%Y%m%d%H'))

    st.line_chart(data,y='T2M')
    print(data)  

############Cooling regression Model###########################
    st.header('Cooling Energy Use by Each AHU (MJ)')

    df_model_cooling = pd.merge(df_Clg_3, data['T2M'], left_index=True, right_index=True)
    # df_model_cooling = pd.merge(df_model_cooling,df, left_index=True, right_index=True)
    predictors = df_model_cooling.drop(['Jean Talon- Chilled Water (MJ)'], axis=1)
    response = df_model_cooling['Jean Talon- Chilled Water (MJ)']
    print('df_model_cooling =')
    print(df_model_cooling)
    print(predictors.shape[1])
    print('Predictors =')
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
    b = (0.01,100)
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

    df_Clg_ahu = pd.DataFrame(columns = df_valves.columns)

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
    kpi_clg = pd.DataFrame({"category":[kpi_clg_ahu.index],"value":[kpi_clg_ahu.values]})
    print(kpi_clg)
    # fig = plt.figure(figsize = (10, 5))
    # merged_list = kpi_clg_ahu+cooling_perimeter_other
    # creating the bar plot
    # plt.bar(m.index, merged_list.values,
    #         width = 0.4)
    ########### Bar chart ########################
    # print(kpi_clg_ahu)
    # print(kpi_clg_ahu.values)
    # # "Energy Costs By Month"
    source = pd.DataFrame()
    source.index = kpi_clg_ahu.index
    source['AHU Cooling Energy(MJ)'] =kpi_clg_ahu.values
    st.bar_chart(source)
    # st.bar_chart(kpi_clg_ahu)
    # data
    #################################################
#     label = ["Cooling Energy Use", 'AH10', 'AH14', 'AH2',
#    'AH8', 'AH9', 'Pent_Elev',
#    'AH7', 'AH16', 'AH11',
#    'AH4', 'AH15', 'AH5',
#    'AH12', 'AH13']
#     source = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#     value = kpi_clg_ahu.values
#     # data to dict, dict to sankey
#     link = dict(source = source, target = target, value = value)
#     node = dict(label = label, pad=50, thickness=7)
#     data = go.Sankey(link = link, node=node)
#     # plot
#     fig = go.Figure(data)
#     fig

    # base = alt.Chart(kpi_clg_ahu).encode(alt.Theta("value:Q").stack(True),alt.Color("category:N").legend(None))
    # pie = base.mark_arc(outerRadius=120)
    # text = base.mark_text(radius=140, size=20).encode(text="category:N")
    # st.altair_chart(base)
    # params = {"ytick.color" : "w",
    #   "xtick.color" : "w",
    #   "axes.labelcolor" : "w",
    #   "axes.edgecolor" : "w"}
    # plt.rcParams.update(params)
    # explode = (0, 0.1, 0, 0,0,0,0.2,0,0,0,0,0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    # ax1.pie(kpi_clg_ahu.values, explode=explode, labels=kpi_clg_ahu.index, autopct='%1.1f%%',
    # shadow=False, startangle=45)
    # ax1.axis('equal') 
    # fig1.patch.set_facecolor('black') # Equal aspect ratio ensures that pie is drawn as a circle.
    # fig1.patch.set_ec('white')
    # st.pyplot(fig1)

    # patches, texts, pcts = ax1.pie(
    #     kpi_clg_ahu.values, labels=kpi_clg_ahu.index, autopct='%.1f%%',
    #     wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
    #     textprops={'size': 'x-large'},
    #     startangle=45)
    # # For each wedge, set the corresponding text label color to the wedge's
    # # face color.
    # for i, patch in enumerate(patches):
    #     texts[i].set_color(patch.get_facecolor())
    #     plt.setp(pcts, color='white')
    # plt.setp(texts)
    # ax1.set_title('Cooling energy use by AHUs %')
    # plt.tight_layout()
    # fig1.patch.set_facecolor('black')
    label_ahu=['AH10', 'AH14', 'AH2',
    'AH8', 'AH9', 'Pent_Elev',
    'AH7', 'AH16', 'AH11',
    'AH4', 'AH15', 'AH5',
    'AH12', 'AH13']
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    for i in range(14)]
    patches, texts = plt.pie(kpi_clg_ahu.values,colors=colors,startangle=90, labels=label_ahu)

    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(label_ahu,100.*kpi_clg_ahu.values/kpi_clg_ahu.values.sum())]
    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1), fontsize=6)
    # fig1.patch.set_facecolor('black')
    plt.ylabel("")
    st.pyplot(fig1)