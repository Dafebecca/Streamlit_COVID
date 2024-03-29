import streamlit as st
st.header("hello world")

# LINK = 'https://www.dropbox.com/scl/fi/kaxqxxhavyb4imp29l2fe/cvd-19-2123.csv?rlkey=s27i5an9f7qw7vuqrdc519hcy&dl=1'
LINK = "https://www.dropbox.com/scl/fi/bszqwd1810ew88xa3gouk/cvd-19-dt-5.csv?rlkey=hsshxu2yr8yheocm24jnqw7ru&dl=1"

import pandas as pd

# Mengubah tautan Dropbox untuk mendownload secara langsung
# Membaca dataset ke dalam pandas DataFrame
df = pd.read_csv(LINK)
# df = df.dropna()
# Menampilkan beberapa baris pertama dari dataframe untuk verifikasi
df['Terinfeksi'] = pd.to_numeric(df['Terinfeksi'], errors='coerce')
df['Sembuh'] = pd.to_numeric(df['Sembuh'], errors='coerce')
df['Meninggal'] = pd.to_numeric(df['Meninggal'], errors='coerce')
df['Total Kasus'] = pd.to_numeric(df['Total Kasus'], errors='coerce')

# Konversi kolom 'Tanggal' menjadi datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
# Mengatur kolom 'Tanggal' sebagai DateTimeIndex
df = df.set_index('Tanggal')
st.dataframe(df)
st.divider()
df.info()

import pandas as pd
import numpy as np
from scipy import interpolate

def impute_mean_median(dataframe, method='mean'):
    """
    Imputasi missing values dengan mean atau median.

    :param dataframe: Pandas DataFrame yang mengandung missing values.
    :param method: 'mean' untuk imputasi dengan mean, 'median' untuk median.
    :return: DataFrame dengan missing values yang diimputasi.
    """
    if method == 'median':
        return dataframe.fillna(dataframe.median())
    return dataframe.fillna(dataframe.mean())

def impute_locf(dataframe):
    """
    Imputasi missing values dengan metode Last Observation Carried Forward (LOCF).

    :param dataframe: Pandas DataFrame yang mengandung missing values.
    :return: DataFrame dengan missing values yang diimputasi.
    """
    return dataframe.fillna(method='ffill')

def impute_linear_interpolation(dataframe):
    """
    Imputasi missing values dengan interpolasi linear.

    :param dataframe: Pandas DataFrame yang mengandung missing values.
    :return: DataFrame dengan missing values yang diimputasi.
    """
    return dataframe.interpolate(method='linear')

def impute_time_series_interpolation(dataframe, method='time'):
    """
    Imputasi missing values dengan metode interpolasi spesifik untuk time series.

    :param dataframe: Pandas DataFrame yang mengandung missing values.
    :param method: Metode interpolasi, default adalah 'time'.
    :return: DataFrame dengan missing values yang diimputasi.
    """
    return dataframe.interpolate(method=method)

# Contoh penggunaan
# df = pd.DataFrame(...)  # Anda harus mengganti ini dengan DataFrame Anda.
st.write("HASIL PREPROCESSING")
df_imputed_ts_int =impute_time_series_interpolation(df)
st.dataframe(df_imputed_ts_int)

with st.sidebar:
    with st.echo():
        st.write("This code will be printed to the sidebar.")
