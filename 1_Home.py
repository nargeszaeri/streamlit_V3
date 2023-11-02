import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title("Energy disaggregation web-application for commercial buildings")
st.write()
st.sidebar.success("Select a page above.")

st.header('Overview')

st.markdown("""This integrated web application offers a user-friendly platform for energy disaggregation analysis.
The application is stored in a public GitHub repository [click here](https://github.com/nargeszaeri/Streamlit_webApp), facilitating access
and collaboration among researchers and practitioners.
The application comprises three pages that enable users to analyze
energy consumption patterns and optimize energy usage.\n""")
  
st.markdown("## Web interface:")

st.markdown("### Page 1: Energy Disaggregation Using BAS Data")
st.write("""The first page of the web application enables users to manually upload BAS and meter data for energy disaggregation
analysis. This feature enhances the flexibility of the application, allowing users to perform energy consumption analysis
even when direct access to BAS data is unavailable. By inputting the necessary data, users can obtain a comprehensive
view of energy consumption attributed to various components within the building, including AHUs, perimeter heating devices, and unmonitored cooling energy consumers.""")  

st.markdown("### Page 2: Energy Disaggregation Using Time series Decomposition ")
st.write("""Unsupervised disaggregation analysis with bulk
meter data. In scenarios where only bulk meter data is
available, the second page of the web application utilizes unsupervised disaggregation techniques, specifically time series
decomposition. This approach disentangles the electricity
consumption patterns into major end-uses. Users can identify
energy anomalies, such as after-hour or overnight energy
waste, and assess the need for rescheduling. The application provides weekly and daily classic time series analysis,
enabling users to gain insights into energy consumption
variations over time.""")
st.markdown("### Page 3: Energy Disaggregation Using BAS from API." )
st.write(""" The third
page of the web application focuses on energy disaggregation
analysis using BAS data. By accessing BAS data and integrating an API for retrieving information from bulk steam
and chilled water meters, the application employs a multiple
linear regression model. This model incorporates outdoor
air temperature to calculate the contribution of each AHU,
perimeter heating device, and hot water consumption. The
outputs of this analysis provide users with detailed insights
into the energy consumption patterns of individual components.""")

         
def main():
    st.header("Publications")
    references = [
        """1- Zaeri, Narges, Araz Ashouri, H. Burak Gunay, and Tareq Abuimara (2022). Disaggregation of
          electricity and heating consumption in commercial buildings with building automation system data
        . In Proceedings of the 9th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, pp. 373-377.""",
        """2- Zaeri, Narges, H. Burak Gunay, and Araz Ashouri. (2022). Unsupervised energy disaggregation using 
        time series decomposition for commercial buildings.""",
        """3- Zaeri Esfahani, Narges, H., Burak Gunay, Araz, Ashouri, Ian, Wilton.
          “End-use disaggregation in commercial buildings with the building automation system trend data”,
            Proceedings of Building Simulation 2021: 17th Conference of IBPS, Bruges, 
            Belgium, 978-1-7750520-2-9, 401-408."""
          ,"""4- Zaeri Esfahani, Narges, Burak Gunay, Araz Ashouri, and Farzeen Rizvi. 
          "An inquiry into the effect of thermal energy meter density and configuration on
            load disaggregation accuracy." Science and Technology for the Built Environment 
            just-accepted (2023): 1-19.ansportation, pp. 373-377."""
        # Add more references as needed
    ]
    
    for reference in references:
        st.write(reference)

if __name__ == "__main__":
    main()
