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

# Initialize session state for My Team page
if "squad" not in st.session_state:
    st.session_state.squad = []
if "budget" not in st.session_state:
    st.session_state.budget = 100.0

# Multi-page navigation
predictions_page = st.Page("pages/1_Predictions.py", title="Predictions", icon="\U0001f4ca", default=True)
my_team_page = st.Page("pages/2_My_Team.py", title="My Team", icon="\u26bd")

pg = st.navigation([predictions_page, my_team_page])

# Mobile bottom tab bar — highlights current page via JS
st.markdown("""
<div class="mobile-nav" id="mobile-nav">
    <a href="/Predictions" id="nav-predictions">
        <span class="nav-icon">📊</span>Predictions
    </a>
    <a href="/My_Team" id="nav-my-team">
        <span class="nav-icon">⚽</span>My Team
    </a>
</div>
<script>
(function() {
    var path = window.location.pathname;
    var links = document.querySelectorAll('#mobile-nav a');
    links.forEach(function(a) {
        if (path === a.getAttribute('href')
            || (path === '/' && a.getAttribute('href') === '/Predictions')) {
            a.classList.add('active');
        }
    });
})();
</script>
""", unsafe_allow_html=True)

pg.run()
