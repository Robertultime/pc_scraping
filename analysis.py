import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import streamlit as st
import note

df = note.get_pc_noted()

max_p = st.slider('x', max_value=df['Price'].max(), min_value=df['Price'].min())  # 👈 this is a widget

disp_df = df[df.Price <= max_p].sort_values('Note_Finale', ascending = False).iloc[:5,:]

st.table(disp_df)

