import streamlit as st
from pages import h, a, b, c

PAGES = {
	"Home": h, 
    "Alzheimers": a,
    "SLI": b,
    "Respiratory Diseases": c,
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()