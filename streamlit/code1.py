import streamlit as st

st.title(" The App Created by Dhilleswar")

st.write("Welcome! this app calculates the sruare of the numbers. ")

st.header(" select a number")
number = st.slider("pick a number",0,100,5)

st.subheader("Result")
squared_number = number * number
st.write(f"the square of **{number}** is **{squared_number}**.")

