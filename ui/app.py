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

# Custom hamburger button — toggles sidebar via JS
st.markdown("""
<div class="hamburger-btn" id="hamburger-btn" onclick="toggleSidebar()">
    <div class="bar"></div>
</div>
<script>
function toggleSidebar() {
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    var isOpen = sidebar && sidebar.getAttribute('aria-expanded') === 'true';

    if (isOpen) {
        // Find close button inside sidebar
        var closeBtn = sidebar.querySelector('[data-testid="stBaseButton-header"]')
            || sidebar.querySelector('button');
        if (closeBtn) {
            closeBtn.style.pointerEvents = 'auto';
            closeBtn.click();
            closeBtn.style.pointerEvents = '';
        }
    } else {
        // Find open button (collapsed control)
        var openBtn = document.querySelector('[data-testid="collapsedControl"] button')
            || document.querySelector('[data-testid="collapsedControl"]');
        if (openBtn) {
            openBtn.style.pointerEvents = 'auto';
            openBtn.click();
            openBtn.style.pointerEvents = '';
        }
    }
}
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
