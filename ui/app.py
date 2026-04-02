"""
FPL Points Predictor — Multi-Page Dashboard

Run with: streamlit run ui/app.py
"""

import os
import sys

# Ensure project root is on the path so `from ui.xxx` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="FPL Points Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject global CSS
from ui.styles import CSS

st.markdown(CSS, unsafe_allow_html=True)

# Custom hamburger button — toggles sidebar by manipulating DOM directly
st.markdown("""
<div class="hamburger-btn" id="hamburger-btn">
    <div class="bar"></div>
</div>
<script>
document.getElementById('hamburger-btn').addEventListener('click', function(e) {
    e.stopPropagation();
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return;
    var isOpen = sidebar.getAttribute('aria-expanded') === 'true';
    sidebar.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
});

// Also close sidebar when clicking the main content area (outside sidebar)
document.addEventListener('click', function(e) {
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    var hamburger = document.getElementById('hamburger-btn');
    if (!sidebar || sidebar.getAttribute('aria-expanded') !== 'true') return;
    if (!sidebar.contains(e.target) && !hamburger.contains(e.target)) {
        sidebar.setAttribute('aria-expanded', 'false');
    }
});
</script>
""", unsafe_allow_html=True)

# Initialize session state for My Team page
if "squad" not in st.session_state:
    st.session_state.squad = []
if "budget" not in st.session_state:
    st.session_state.budget = 100.0

# Multi-page navigation
predictions_page = st.Page("pages/1_Predictions.py", title="Predictions", icon="\U0001f4ca", default=True)
my_team_page = st.Page("pages/2_My_Team.py", title="My Team", icon="\u26bd")

pg = st.navigation([predictions_page, my_team_page])

pg.run()
