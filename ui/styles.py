"""
FPL-themed CSS styles for the dashboard.
"""

# FPL Color Palette
PRIMARY = "#37003c"       # FPL purple
ACCENT = "#00ff87"        # FPL green
TEAL = "#02efbc"
BG_LIGHT = "#f8f9fa"
WHITE = "#ffffff"
TEXT_PRIMARY = "#37003c"
TEXT_SECONDARY = "#666666"

# FDR colors
FDR_COLORS = {
    1: "#147d1b",  # Dark green - very easy
    2: "#00cc2b",  # Green - easy
    3: "#999999",  # Gray - medium
    4: "#ff6200",  # Orange - hard
    5: "#d40000",  # Red - very hard
}

FDR_BG_COLORS = {
    1: "#d4edda",
    2: "#e8f5e9",
    3: "#f0f0f0",
    4: "#fff3e0",
    5: "#ffebee",
}

CSS = """
<style>
    /* ---- Global ---- */
    .block-container {
        padding-top: 1rem;
    }

    /* ---- Header ---- */
    .fpl-header {
        background: linear-gradient(135deg, #37003c 0%, #2d0033 60%, #00ff87 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 20px;
        text-align: center;
    }
    .fpl-header h1 {
        color: white;
        font-size: 2.2em;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .fpl-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.05em;
        margin: 0;
    }

    /* ---- Metric Cards ---- */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
    }
    .metric-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        flex: 1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card .label {
        color: #888;
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.6em;
        font-weight: 800;
        color: #37003c;
    }
    .metric-card .sub {
        color: #999;
        font-size: 0.8em;
        margin-top: 2px;
    }

    /* ---- Filter Bar ---- */
    .filter-bar {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 16px 20px 8px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ---- FDR Badges ---- */
    .fdr-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82em;
        font-weight: 600;
        margin: 1px 2px;
        line-height: 1.4;
    }
    .fdr-1 { background: #d4edda; color: #147d1b; }
    .fdr-2 { background: #e8f5e9; color: #1b8a2a; }
    .fdr-3 { background: #f0f0f0; color: #666; }
    .fdr-4 { background: #fff3e0; color: #e65100; }
    .fdr-5 { background: #ffebee; color: #c62828; }

    /* ---- Home/Away Badges ---- */
    .badge-home {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82em;
    }
    .badge-away {
        background: #fce4ec;
        color: #c62828;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82em;
    }

    /* ---- Position Badges ---- */
    .pos-gk { background: #fff8e1; color: #f9a825; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8em; }
    .pos-def { background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8em; }
    .pos-mid { background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8em; }
    .pos-fwd { background: #fce4ec; color: #c62828; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8em; }

    /* ---- Confidence ---- */
    .conf-high { color: #2e7d32; font-weight: 700; }
    .conf-med { color: #e65100; font-weight: 700; }
    .conf-low { color: #c62828; font-weight: 700; }

    /* ---- Player Cards (My Team) ---- */
    .player-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }
    .player-card .name {
        font-weight: 700;
        font-size: 0.9em;
        color: #37003c;
        margin-bottom: 2px;
    }
    .player-card .info {
        color: #888;
        font-size: 0.75em;
    }
    .player-card .pts {
        font-size: 1.3em;
        font-weight: 800;
        color: #37003c;
        margin: 4px 0;
    }

    /* ---- Transfer Card ---- */
    .transfer-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .transfer-out {
        color: #c62828;
        font-weight: 600;
    }
    .transfer-in {
        color: #2e7d32;
        font-weight: 600;
    }
    .transfer-arrow {
        color: #37003c;
        font-size: 1.2em;
        font-weight: 800;
        margin: 0 8px;
    }
    .delta-positive { color: #2e7d32; font-weight: 700; }
    .delta-negative { color: #c62828; font-weight: 700; }

    /* ---- Budget Bar ---- */
    .budget-bar {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .budget-label {
        color: #888;
        font-size: 0.8em;
        text-transform: uppercase;
    }
    .budget-value {
        font-size: 1.3em;
        font-weight: 800;
        color: #37003c;
    }

    /* ---- Squad Pitch Row ---- */
    .pitch-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 8px;
        flex-wrap: wrap;
    }

    /* ---- Section Headers ---- */
    .section-header {
        color: #37003c;
        font-size: 1.3em;
        font-weight: 700;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #00ff87;
        display: inline-block;
    }

    /* ---- FDR Legend ---- */
    .fdr-legend {
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 0.8em;
        color: #888;
        margin-bottom: 12px;
    }
    .fdr-legend-item {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .fdr-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }

    /* ---- Table Overrides ---- */
    div[data-testid="stDataFrame"] table {
        font-size: 0.88em;
    }
    div[data-testid="stDataFrame"] th {
        background: #37003c !important;
        color: white !important;
        font-weight: 600 !important;
    }

    /* ---- Navigation Styling ---- */
    [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }

    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #37003c;
    }
</style>
"""
