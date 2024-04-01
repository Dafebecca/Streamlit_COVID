# Streamlit --> It helps to create interactive websites
import streamlit as st

# Pandas --> It helps manipulate data
import pandas as pd

# Matplotlib --> It helps create visualizations
import matplotlib.pyplot as plt

# Numpy --> It helps with calculations
import numpy as np

# Warnings --> It helps to display a warning message
import warnings

# Datetime --> It helps to process dates and times
from datetime import datetime

# Solve Ivp --> It helps to solve differential equations
from scipy.integrate import solve_ivp

# Minimize --> It helps to assist in optimization
from scipy.optimize import minimize

########################################## TITLE #############################################
##############################################################################################

st.set_page_config(layout="wide")
st.header("Data Modeling COVID Bandung (2021-2023): Runge Kutta")

######################################## LOAD DATA ###########################################
##############################################################################################

# Change the Dropbox link to download directly
LINK = "https://www.dropbox.com/scl/fi/bszqwd1810ew88xa3gouk/cvd-19-dt-5.csv?rlkey=hsshxu2yr8yheocm24jnqw7ru&dl=1"

# Read the dataset into a Pandas DataFrame
df = pd.read_csv(LINK)

######################################## FORMAT DATA #########################################
##############################################################################################

# Change the data type for the infected, recovered, dead, and total cases columns to numeric
df["Terinfeksi"] = pd.to_numeric(df["Terinfeksi"], errors="coerce")
df["Sembuh"] = pd.to_numeric(df["Sembuh"], errors= "coerce")
df["Meninggal"] = pd.to_numeric(df["Meninggal"], errors="coerce")
df["Total Kasus"] = pd.to_numeric(df["Total Kasus"], errors="coerce")

# Convert the "Date" column to datetime
df["Tanggal"] = pd.to_datetime(df["Tanggal"])

# Set the "Date" column as DateTimeIndex
df = df.set_index("Tanggal")
df.index = pd.to_datetime(df.index)

################################## HANDLING MISSING VALUES ###################################
##############################################################################################

# LOCF (Last Observation Carried Forward)
df = df.fillna(method='ffill')

########################################## FUNCTION ###########################################
###############################################################################################

def sample_data_by_date_range(df, start_date, end_date):
    """
    Samples data from a DataFrame based on a specific date range

    Parameters:
    df (pd.DataFrame): DataFrame containing data.
    start_date (str): Start date (inclusive), in format "YYYY-MM-DD HH:MM:SS"
    end_date (str): End date (inklusif), in format "YYYY-MM-DD HH:MM:SS"

    Returns:
    pd.DataFrame: A DataFrame containing data from a specified date range
    """
    # Make sure the date column is of type datetime
    df.index = pd.to_datetime(df.index)
    
    # Time-based indexing to retrieve data within a date range
    return df[start_date:end_date]

################################## VISUALIZATION SELECTED DATA ################################
###############################################################################################

def visualize_covid(df_selected,label_title):
    fig, ax2 = plt.subplots(figsize=(20, 8))
    ax2.plot(df_selected.index,df_selected["Terinfeksi"], color="red")
    ax2.set_title(f"Selected COVID-19 Data Range {start_date} -  {end_date}")
    ax2.set_xticks(df_selected.index)
    ax2.set_xticklabels(df_selected.index, rotation=90)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Cases")
    st.write(fig)

##################################### SIR MODEL ###############################################
###############################################################################################

def sir_model(t, y, beta, gamma):
    S, I, R = y
    beta = _beta
    gamma = _gamma
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to calculate total infected and recovered cases
def total_infected(t, beta, gamma):
    I0 = data[0]
    R0 = 0
    S0 = N - I0 - R0
    sol = solve_ivp(sir_model, [t.min(), t.max()], [S0, I0, R0], args=(beta, gamma), dense_output=True)
    return sol.sol(t)[1] + sol.sol(t)[2]  # I + R

def daily_infected(t, beta, gamma):
    I0 = data[0]
    R0 = 0
    S0 = N - I0 - R0
    sol = solve_ivp(sir_model, [t.min(), t.max()], [S0, I0, R0], args=(beta, gamma), dense_output=True)
    
    # Infected
    I = sol.sol(t)[1]

    # Calculates daily changes in infected cases
    daily_cases = np.diff(I, prepend=I0)  
    return daily_cases

# Cost function for optimization
def cost_function_daily_infected(params):
    beta, gamma = params
    predicted = daily_infected(t, beta, gamma)
    return np.sum((predicted - data)**2)

# Cost function for optimization
def cost_function(params):
    beta, gamma = params
    predicted = total_infected(t, beta, gamma)
    return np.sum((predicted - data)**2)

##################################### CONFIGURASI MODEL #######################################
###############################################################################################

# Create data categories
st.sidebar.subheader("Category")
pilihan_kategori = st.sidebar.selectbox("Select Category",
                     options=["Terinfeksi","Sembuh","Meninggal","Total Kasus"])

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(df.index,df[pilihan_kategori], color="skyblue")
ax.set_title("Dataset - COVID-19 Bandung")
st.write(fig)

st.sidebar.divider()
st.sidebar.subheader("Data Range")

options_date = df.index
counter =  len(options_date)-1 

start_date = st.sidebar.date_input("Start Period",value=datetime.strptime('2021-05-01', '%Y-%m-%d').date())
end_date =  st.sidebar.date_input("End Period", value=datetime.strptime('2021-07-28', '%Y-%m-%d').date())


if st.sidebar.button("Select Data"):
    ne = sample_data_by_date_range(df, start_date=start_date, end_date=end_date)
    visualize_covid(ne,f'Visualisasi Data Covid-19 {start_date} - {end_date}')

st.sidebar.divider()
st.sidebar.subheader("SIR Runge-Kutta Modeling")

_initialP = st.sidebar.number_input("Initial Population", value=8000)
_beta = st.sidebar.text_input("Parameter Beta", value=0.001)
_gamma = st.sidebar.text_input("Parameter Gamma", value=0.001)
_beta = float(_beta)
_gamma = float(_gamma)
_optimizer = st.sidebar.checkbox("Automate Search Paramter")

if st.sidebar.button(label="Running",type="primary"):
    data = df[start_date:end_date]["Terinfeksi"].to_numpy()
    I0 =  data[0]
    N =  _initialP
    R0 = 0
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]
    t = np.arange(len(data))
    if _optimizer == True:
       initial_guess = [0.01, 1/10]
       result = minimize(cost_function_daily_infected, initial_guess, method="L-BFGS-B", bounds=[(0, 1), (0, 1)])
       beta_opt, gamma_opt = result.x
       st.sidebar.write(f"Beta Value: {format(beta_opt,'.5f')}, Gamma Value: {format(gamma_opt,'.5f')}")
       sol = solve_ivp(sir_model, [t[0], t[-1]], y0, args=(beta_opt, gamma_opt), t_eval=t, vectorized=True)
    else:
        # Solving SIR models using RK4 (Scipy's solve_ivp uses RK45 by default)
        sol = solve_ivp(sir_model, [t[0], t[-1]], y0, args=(_beta, _gamma), t_eval=t, vectorized=True)

with st.container(border=True):
     st.text("Analysis")

     try:
            # Plotting actual and predicted data
        plt.figure(figsize=(14, 7))
        plt.plot(df[start_date:end_date].index, df[start_date:end_date]["Terinfeksi"], label="Actual Data on Infected Cases")
        plt.plot(df[start_date:end_date].index, sol.y[1] + sol.y[2], label="SIR Model Prediction (RK4)")
        plt.xlabel("Day")
        plt.xticks(df[start_date:end_date].index,rotation=90)
        plt.ylabel("Total Cases")
        plt.title("Comparison of Actual COVID-19 Cases with SIR Model Predictions")
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)
     except Exception as e:
            st.success("No Data Available")
