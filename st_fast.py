import streamlit as st
import note


pc_noted = note.get_pc_noted()
df = note.get_best_pc(pc_noted)
st.table(df)