import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np

import pystan
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import json
from fbprophet.serialize import model_to_json, model_from_json

import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools
from datetime import datetime
import json

st.set_page_config(page_title ="FBProphet Forecasting App",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")


tabs = ["Application","About"]
page = st.sidebar.radio("Tabs",tabs)


@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)
def load_csv():
    
    df_input = pd.DataFrame()  
    df_input=pd.read_csv(input,sep=None ,engine='python', encoding='utf-8',
                            parse_dates=True,
                            infer_datetime_format=True)
    return df_input

def prep_data(df):

    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("The selected date column is now labeled as **ds** and the values columns as **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input

code1 = """                       
st.dataframe(df)
                       
st.write(df.describe())

try:
    line_chart = alt.Chart(df).mark_line().encode(
    x = 'ds:T',
    y = "y:Q").properties(title="Time series preview").interactive()
        st.altair_chart(line_chart,use_container_width=True)
except:
    st.line_chart(df['y'],use_container_width =True,height = 300) """ 
code2="""
 m = Prophet(
    seasonality_mode=seasonality,
    daily_seasonality=daily,
    weekly_seasonality=weekly,
    yearly_seasonality=yearly,
    growth=growth,
    changepoint_prior_scale=changepoint_scale,
    seasonality_prior_scale= seasonality_scale)
if holidays:
    m.add_country_holidays(country_name=selected_country)
                        
if monthly:
    m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)

m = m.fit(df)
future = m.make_future_dataframe(periods=periods_input,freq='D')
future['cap']=cap
future['floor']=floor
                """

if page == "Application":
    
    with st.sidebar:
        if st.button(label='Clear cache'):
            caching.clear_cache()

    st.title('Forecast application 🧙🏻')
    st.write('This app enables you to generate time series forecast withouth any dependencies.')
    st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")
    caching.clear_cache()
    df =  pd.DataFrame()   

    st.subheader('1. Data loading 🏋️')
    st.write("Import a time series csv file.")
    with st.beta_expander("Data format"): 
        st.write("The dataset can contain multiple columns but you will need to select a column to be used as dates and a second column containing the metric you wish to forecast. The columns will be renamed as **ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric.")

    input = st.file_uploader('')
    
    if input is None:
        st.write("Or use sample dataset to try the application")
        sample = st.checkbox("Download sample data from GitHub")

    try:
        if sample:
            st.markdown("""[download_link](https://gist.github.com/jhamuza/b8b285b992afed4f479e01fcb4c1e88a)""")    
            
    except:

        if input:
            with st.spinner('Loading data..'):
                df = load_csv()
        
                st.write("Columns:")
                st.write(list(df.columns))
                columns = list(df.columns)
        
                col1,col2 = st.beta_columns(2)
                with col1:
                    date_col = st.selectbox("Select date column",index= 0,options=columns,key="date")
                with col2:
                    metric_col = st.selectbox("Select values column",index=1,options=columns,key="values")

                df = prep_data(df)
                output = 0
    

        if st.checkbox('Chart data',key='show'):
            with st.spinner('Plotting data..'):
                col1,col2 = st.beta_columns(2)
                with col1:
                    st.dataframe(df)
                    
                with col2:    
                    st.write("Dataframe description:")
                    st.write(df.describe())

            try:
                line_chart = alt.Chart(df).mark_line().encode(
                    x = 'ds:T',
                    y = "y:Q",tooltip=['ds:T', 'y']).properties(title="Time series preview").interactive()
                st.altair_chart(line_chart,use_container_width=True)
                
            except:
                st.line_chart(df['y'],use_container_width =True,height = 300)
                
            

    st.subheader("2. Parameters configuration 🛠️")

    with st.beta_container():
        st.write('In this section you can modify the algorithm settings.')
            
        with st.beta_expander("Horizon"):
            periods_input = st.number_input('Select how many future periods (days) to forecast.',
            min_value = 1, max_value = 366,value=90)

        with st.beta_expander("Seasonality"):
            st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required. For more informations visit the [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)""")
            seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

        with st.beta_expander("Trend components"):
            st.write("Add or remove components:")
            daily = st.checkbox("Daily")
            weekly= st.checkbox("Weekly")
            monthly = st.checkbox("Monthly")
            yearly = st.checkbox("Yearly")
        
        with st.beta_expander("Growth model"):
            st.write('Prophet uses by default a linear growth model.')
            st.markdown("""For more information check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)""")

            growth = st.radio(label='Growth model',options=['linear',"logistic"]) 

            if growth == 'linear':
                growth_settings= {
                            'cap':1,
                            'floor':0
                        }
                cap=1
                floor=1
                df['cap']=1
                df['floor']=0

            if growth == 'logistic':
                st.info('Configure saturation')

                cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
                floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
                if floor > cap:
                    st.error('Invalid settings. Cap must be higher then floor.')
                    growth_settings={}

                if floor == cap:
                    st.warning('Cap must be higher than floor')
                else:
                    growth_settings = {
                        'cap':cap,
                        'floor':floor
                        }
                    df['cap']=cap
                    df['floor']=floor
        
        with st.beta_expander('Hyperparameters'):
            st.write('In this section it is possible to tune the scaling coefficients.')
            
            seasonality_scale_values= [0.1, 1.0,5.0,10.0]    
            changepoint_scale_values= [0.01, 0.1, 0.5,1.0]

            st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
            changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
            
            st.write("The seasonality change point controls the flexibility of the seasonality.")
            seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)    

            st.markdown("""For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")
                    
                    
    with st.beta_container():
        st.subheader("3. Forecast 🔮")
        st.write("Fit the model on the data and generate future prediction.")
        st.write("Load a time series to activate.")
        
        if input:
            
            if st.checkbox("Initialize model (Fit)",key="fit"):
                if len(growth_settings)==2:
                    m = Prophet(seasonality_mode=seasonality,
                                daily_seasonality=daily,
                                weekly_seasonality=weekly,
                                yearly_seasonality=yearly,
                                growth=growth,
                                changepoint_prior_scale=changepoint_scale,
                                seasonality_prior_scale= seasonality_scale)
                  
                        
                    if monthly:
                        m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)

                    with st.spinner('Fitting the model..'):

                        m = m.fit(df)
                        future = m.make_future_dataframe(periods=periods_input,freq='D')
                        future['cap']=cap
                        future['floor']=floor
                        st.write("The model will produce forecast up to ", future['ds'].max())
                        st.success('Model fitted successfully')

                else:
                    st.warning('Invalid configuration')

            if st.checkbox("Generate forecast (Predict)",key="predict"):
                try:
                    with st.spinner("Forecasting.."):

                        forecast = m.predict(future)
                        st.success('Prediction generated successfully')
                        st.dataframe(forecast)
                        fig1 = m.plot(forecast)
                        st.write(fig1)
                        output = 1

                        if growth == 'linear':
                            fig2 = m.plot(forecast)
                            a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                            st.write(fig2)
                            output = 1
                except:
                    st.warning("You need to train the model first.. ")
                        
            
            if st.checkbox('Show components'):
                try:
                    with st.spinner("Loading.."):
                        fig3 = m.plot_components(forecast)
                        st.write(fig3)
                except: 
                    st.warning("Requires forecast generation..")  

        st.subheader('6. Export results ✨')
        
        st.write("Finally you can export your result forecast, model configuration and evaluation metrics.")
        
        if input:
            if output == 1:
                col1, col4 = st.beta_columns(2) #test

                with col1:
                    
                    if st.button('Export forecast (.csv)'):
                        with st.spinner("Exporting.."):

                            #export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']]).to_csv()
                            export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']])
                            st.write(export_forecast.head())
                            export_forecast= export_forecast.to_csv(decimal=',')
                            b64 = base64.b64encode(export_forecast.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **forecast.csv**)'
                            st.markdown(href, unsafe_allow_html=True)
                            
                with col4:
                    if st.button('Clear cache memory please'):
                        caching.clear_cache()

            else:
                st.write("Generate a forecast to download.")
            

if page == "About":
    st.image("prophet.png")
    st.header("About")
    st.markdown("Official documentation of **[Facebook Prophet](https://facebook.github.io/prophet/)**")
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.write("Author:")
    st.markdown(""" **[Giancarlo Di Donato](https://www.linkedin.com/in/giancarlodidonato/)**""")
    st.markdown("""**[Source code](https://github.com/giandata/forecast-app)**""")
    st.write("Deployed for Capstone Project by:")
    st.markdown(""" **[Hamzah](https://github.com/jhamuza)**""")
    st.markdown("""**[Source code](https://github.com/jhamuza/testing-prophet)**""")

    st.write("Created on 27/02/2021")
    st.write("Last updated: 29/04/2021")
    st.write("Amended on: **12/03/2022**")
