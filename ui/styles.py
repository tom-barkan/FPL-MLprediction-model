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

# Team colors for dot on pitch cards (keyed by short_name from CSV)
TEAM_COLORS = {
    "ARS": "#EF0107", "AVL": "#95BFE5", "BOU": "#DA291C",
    "BRE": "#e30613", "BHA": "#0057B8", "CHE": "#034694",
    "CRY": "#1B458F", "EVE": "#003399", "FUL": "#000000",
    "LEE": "#005DAA", "LIV": "#C8102E",
    "MCI": "#6CABDD", "MUN": "#DA291C", "NEW": "#241F20",
    "NFO": "#DD0000", "SUN": "#EB172B", "TOT": "#132257",
    "WHU": "#7A263A", "WOL": "#FDB913", "BUR": "#6C1D45",
}

CSS = """
<style>
    /* ---- Global ---- */
    /* Make header transparent — removes white bar but keeps toggle button */
    [data-testid="stHeader"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        pointer-events: none;
    }
    /* Re-enable clicks on the sidebar toggle inside the header */
    [data-testid="stHeader"] [data-testid="collapsedControl"] {
        pointer-events: auto;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stApp {
        background: #f5f0f7;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #37003c !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        color: rgba(255,255,255,0.85) !important;
        border-radius: 8px;
        padding: 8px 12px;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(255,255,255,0.12) !important;
        color: #00ff87 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] {
        background: rgba(0,255,135,0.15) !important;
        color: #00ff87 !important;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 1.5rem;
    }

    /* ---- Header Banner ---- */
    .fpl-header {
        background: linear-gradient(135deg, #37003c 0%, #4a0e50 40%, #1a6840 80%, #00ff87 100%);
        color: white;
        padding: 32px 36px;
        border-radius: 16px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(55,0,60,0.25);
    }
    .fpl-header h1 {
        color: white;
        font-size: 2.4em;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .fpl-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1em;
        margin: 0;
        font-weight: 400;
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
        padding: 18px 20px;
        text-align: center;
        flex: 1;
        box-shadow: 0 2px 8px rgba(55,0,60,0.06);
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(55,0,60,0.12);
    }
    .metric-card .label {
        color: #888;
        font-size: 0.72em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.7em;
        font-weight: 800;
        color: #37003c;
        line-height: 1.2;
    }
    .metric-card .sub {
        color: #aaa;
        font-size: 0.78em;
        margin-top: 4px;
    }

    /* ---- Filter Bar ---- */
    .filter-bar {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 16px 20px 8px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(55,0,60,0.06);
    }

    /* ---- Position Toggle Pills ---- */
    .pos-pills {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .pos-pill {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        cursor: pointer;
        transition: all 0.15s;
        border: 2px solid transparent;
    }
    .pos-pill-gk { background: #fff8e1; color: #f9a825; border-color: #f9a825; }
    .pos-pill-def { background: #e3f2fd; color: #1565c0; border-color: #1565c0; }
    .pos-pill-mid { background: #e8f5e9; color: #2e7d32; border-color: #2e7d32; }
    .pos-pill-fwd { background: #fce4ec; color: #c62828; border-color: #c62828; }
    .pos-pill-active {
        background: #00ff87 !important;
        color: #37003c !important;
        border-color: #00ff87 !important;
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

    /* ---- Player Cards (generic) ---- */
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
        border-bottom: 3px solid #00ff87;
        display: inline-block;
    }

    /* ---- FDR Legend Strip ---- */
    .fdr-legend {
        display: flex;
        gap: 14px;
        align-items: center;
        font-size: 0.8em;
        color: #888;
        margin-bottom: 16px;
        background: white;
        padding: 10px 18px;
        border-radius: 10px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 1px 4px rgba(55,0,60,0.04);
    }
    .fdr-legend-label {
        font-weight: 700;
        color: #37003c;
        font-size: 0.9em;
    }
    .fdr-legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
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
        font-size: 0.85em !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stDataFrame"] td {
        border-bottom: 1px solid #f0ecf2 !important;
    }
    div[data-testid="stDataFrame"] tr:nth-child(even) td {
        background: #faf8fb !important;
    }
    div[data-testid="stDataFrame"] tr:hover td {
        background: #f0ecf2 !important;
    }

    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-weight: 700;
        color: #37003c;
        font-size: 1.05em;
    }
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(55,0,60,0.06);
        margin-bottom: 12px;
    }

    /* ---- Floating sidebar toggle button ---- */
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 12px;
        left: 12px;
        z-index: 999999 !important;
        background: #37003c !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 12px rgba(55,0,60,0.3);
        border: 2px solid #00ff87 !important;
    }
    [data-testid="collapsedControl"] button {
        color: #00ff87 !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #00ff87 !important;
        stroke: #00ff87 !important;
    }

    /* ---- Footer ---- */
    .fpl-footer {
        text-align: center;
        color: #aaa;
        font-size: 0.82em;
        padding: 16px 0 8px;
        border-top: 1px solid #e8e8e8;
        margin-top: 24px;
    }

    /* ================================================ */
    /* ---- My Team: Pitch Layout ----                  */
    /* ================================================ */
    .pitch-container {
        background: linear-gradient(180deg, #1a6e2e 0%, #2d8c3c 25%, #34993f 50%, #2d8c3c 75%, #1a6e2e 100%);
        border-radius: 16px;
        padding: 24px 8px 18px;
        position: relative;
        overflow: hidden;
        min-height: 480px;
    }
    /* Outer pitch outline */
    .pitch-outline {
        position: absolute;
        top: 14px; left: 8%; right: 8%; bottom: 14px;
        border: 1.5px solid rgba(255,255,255,0.22);
        border-radius: 4px;
        pointer-events: none;
    }
    /* Centre line */
    .pitch-centre-line {
        position: absolute;
        top: 50%; left: 8%; right: 8%;
        height: 0;
        border-top: 1.5px solid rgba(255,255,255,0.22);
        pointer-events: none;
    }
    /* Centre circle */
    .pitch-centre-circle {
        position: absolute;
        top: calc(50% - 40px); left: calc(50% - 40px);
        width: 80px; height: 80px;
        border: 1.5px solid rgba(255,255,255,0.18);
        border-radius: 50%;
        pointer-events: none;
    }
    /* Penalty box top (FWD end) */
    .pitch-box-top {
        position: absolute;
        top: 14px; left: 25%; right: 25%;
        height: 50px;
        border: 1.5px solid rgba(255,255,255,0.16);
        border-top: none;
        pointer-events: none;
    }
    /* Penalty box bottom (GK end) */
    .pitch-box-bottom {
        position: absolute;
        bottom: 14px; left: 25%; right: 25%;
        height: 50px;
        border: 1.5px solid rgba(255,255,255,0.16);
        border-bottom: none;
        pointer-events: none;
    }

    .pitch-pos-label {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.62em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 4px;
        position: relative;
        z-index: 1;
    }
    .pitch-row-flex {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-bottom: 12px;
        position: relative;
        z-index: 1;
        flex-wrap: wrap;
    }

    /* Individual player card ON the pitch */
    .pitch-player {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 7px 8px 5px;
        text-align: center;
        min-width: 84px;
        max-width: 108px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.22);
        flex: 0 0 auto;
    }
    .pitch-player .pp-photo {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        object-fit: cover;
        object-position: top;
        border: 2px solid #e8e8e8;
        margin-bottom: 2px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .pitch-player .pp-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 2px;
        vertical-align: middle;
    }
    .pitch-player .pp-name {
        font-weight: 700;
        font-size: 0.72em;
        color: #1a1a2e;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100px;
    }
    .pitch-player .pp-meta {
        color: #888;
        font-size: 0.6em;
        margin: 1px 0;
    }
    .pitch-player .pp-pts {
        font-size: 1.1em;
        font-weight: 800;
        color: #37003c;
        line-height: 1.3;
    }
    .pitch-player .pp-fix {
        margin-top: 3px;
    }
    .pitch-player .pp-fix .fdr-badge {
        font-size: 0.55em;
        padding: 1px 4px;
        border-radius: 3px;
        margin: 0 1px;
    }

    /* ================================================ */
    /* ---- My Team: Transfer Planner ----              */
    /* ================================================ */
    .tp-section-hdr {
        background: #37003c;
        color: #fff;
        padding: 7px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 14px 0 8px;
    }

    .tp-out-banner {
        background: #fff5f5;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .tp-out-banner .out-pill {
        background: #c62828;
        color: #fff;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78em;
    }
    .tp-out-banner .out-name {
        font-weight: 700;
        color: #1a1a2e;
        font-size: 0.9em;
    }
    .tp-out-banner .out-fixtures {
        margin-left: auto;
    }

    /* Suggestion card */
    .sg-card {
        background: #fff;
        border: 1px solid #eaeaea;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .sg-card .sg-top {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .sg-card .sg-name {
        font-weight: 700;
        font-size: 0.92em;
        color: #1a1a2e;
    }
    .sg-card .sg-pos-badge {
        display: inline-block;
        padding: 1px 7px;
        border-radius: 4px;
        font-size: 0.68em;
        font-weight: 700;
    }
    .sg-card .sg-team-price {
        color: #888;
        font-size: 0.78em;
    }
    .sg-card .sg-stats-row {
        display: flex;
        gap: 14px;
        margin-top: 6px;
        flex-wrap: wrap;
        align-items: center;
    }
    .sg-card .sg-stat {
        text-align: center;
        min-width: 56px;
    }
    .sg-card .sg-stat .s-label {
        font-size: 0.6em;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .sg-card .sg-stat .s-val {
        font-size: 0.88em;
        font-weight: 700;
        color: #37003c;
    }
    .sg-card .sg-stat .s-delta {
        font-size: 0.66em;
    }
    .sg-card .sg-fixtures {
        display: flex;
        gap: 3px;
        align-items: center;
    }
    .sg-card .sg-fixtures .fdr-badge {
        font-size: 0.65em;
        padding: 2px 6px;
    }

    /* ---- Compact metric row for My Team header ---- */
    .mc-row {
        display: flex;
        gap: 8px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }
    .mc-item {
        background: #fff;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
        flex: 1;
        min-width: 70px;
        box-shadow: 0 1px 3px rgba(55,0,60,0.04);
    }
    .mc-item .mc-label {
        color: #888;
        font-size: 0.62em;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 2px;
        font-weight: 600;
    }
    .mc-item .mc-value {
        font-size: 1.2em;
        font-weight: 800;
        color: #37003c;
    }
</style>
"""
