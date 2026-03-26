"""
Reusable HTML UI components for the FPL dashboard.
"""

from ui.styles import FDR_COLORS


def fdr_emoji(fdr):
    """Return FDR as colored circle emojis for use in dataframes."""
    fdr = int(fdr) if fdr else 0
    mapping = {1: "\U0001f7e2", 2: "\U0001f7e2", 3: "\u26aa", 4: "\U0001f7e0", 5: "\U0001f534"}
    return mapping.get(fdr, "")


def fixture_text(opponent, home_away, fdr):
    """Format fixture as readable text for dataframe columns."""
    if not opponent or opponent == "-":
        return "-"
    fdr_indicator = fdr_emoji(fdr)
    return f"{opponent}({home_away}) {fdr_indicator}"


def fixture_badge_html(opponent, home_away, fdr):
    """Render a fixture as a colored HTML badge."""
    if not opponent or opponent == "-":
        return "<span style='color:#ccc;'>-</span>"
    fdr = int(fdr) if fdr else 3
    css_class = f"fdr-{fdr}"
    ha_label = "H" if home_away == "H" else "A"
    return f'<span class="fdr-badge {css_class}">{opponent} ({ha_label})</span>'


def fixture_run_html(fixtures_list):
    """Render a list of fixture dicts as side-by-side badges."""
    badges = []
    for fix in fixtures_list:
        badges.append(fixture_badge_html(fix["opponent"], fix["home_away"], fix["fdr"]))
    return " ".join(badges)


def position_badge(pos):
    """Colored position badge HTML."""
    css_class = {
        "GK": "pos-gk",
        "DEF": "pos-def",
        "MID": "pos-mid",
        "FWD": "pos-fwd",
    }.get(pos, "pos-mid")
    return f'<span class="{css_class}">{pos}</span>'


def confidence_pill(value):
    """Colored confidence display HTML."""
    value = int(value)
    if value >= 75:
        css_class = "conf-high"
    elif value >= 50:
        css_class = "conf-med"
    else:
        css_class = "conf-low"
    return f'<span class="{css_class}">{value}%</span>'


def fdr_legend_html():
    """Render horizontal FDR legend."""
    items = [
        (1, "Very Easy"),
        (2, "Easy"),
        (3, "Medium"),
        (4, "Hard"),
        (5, "Very Hard"),
    ]
    html = '<div class="fdr-legend"><span>Fixture Difficulty:</span>'
    for fdr, label in items:
        color = FDR_COLORS[fdr]
        html += f'<span class="fdr-legend-item"><span class="fdr-dot" style="background:{color};"></span>{label}</span>'
    html += "</div>"
    return html


def player_card_html(name, team, price, pts, fixtures_list=None):
    """Render a player card for the pitch layout."""
    fixtures_str = ""
    if fixtures_list:
        fixtures_str = fixture_run_html(fixtures_list)

    return f"""
    <div class="player-card">
        <div class="name">{name}</div>
        <div class="info">{team} &middot; {price:.1f}m</div>
        <div class="pts">{pts:.1f}</div>
        {f'<div style="margin-top:4px;">{fixtures_str}</div>' if fixtures_str else ''}
    </div>
    """


def metric_card_html(label, value, sub=""):
    """Render a summary metric card."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """


def delta_html(value, suffix="pts"):
    """Render a positive/negative delta value."""
    if value > 0:
        return f'<span class="delta-positive">+{value:.1f} {suffix}</span>'
    elif value < 0:
        return f'<span class="delta-negative">{value:.1f} {suffix}</span>'
    return f'<span style="color:#888;">0.0 {suffix}</span>'
