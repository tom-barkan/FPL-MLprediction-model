"""
Reusable HTML UI components for the FPL dashboard.
"""

from ui.styles import FDR_COLORS, TEAM_COLORS


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
    """Render horizontal FDR legend strip."""
    items = [
        (1, "Very Easy"),
        (2, "Easy"),
        (3, "Medium"),
        (4, "Hard"),
        (5, "Very Hard"),
    ]
    html = '<div class="fdr-legend"><span class="fdr-legend-label">FDR Legend</span>'
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


def pitch_player_html(name, team_name, price, pts, fixtures_list=None):
    """Render a compact player card for the green pitch layout."""
    team_color = TEAM_COLORS.get(team_name, "#37003c")

    fixtures_str = ""
    if fixtures_list:
        mini_badges = []
        for fix in fixtures_list:
            fdr = int(fix["fdr"]) if fix.get("fdr") else 3
            ha = "H" if fix["home_away"] == "H" else "A"
            mini_badges.append(
                f'<span class="fdr-badge fdr-{fdr}">{fix["opponent"]} ({ha})</span>'
            )
        fixtures_str = " ".join(mini_badges)

    return f"""
    <div class="pitch-player">
        <div class="pp-name">
            <span class="pp-dot" style="background:{team_color};"></span>
            {name}
        </div>
        <div class="pp-meta">{team_name} &middot; {price:.1f}m</div>
        <div class="pp-pts">{pts:.1f}</div>
        {f'<div class="pp-fix">{fixtures_str}</div>' if fixtures_str else ''}
    </div>
    """


def pitch_html(position_rows):
    """
    Render a full pitch with players arranged by position.

    position_rows: list of (pos_label, [card_html, ...]) tuples,
    ordered top-to-bottom (FWD first, GK last).
    """
    rows_html = ""
    for pos_label, cards in position_rows:
        cards_joined = "\n".join(cards)
        rows_html += f"""
        <div class="pitch-pos-label">{pos_label}</div>
        <div class="pitch-row-flex">{cards_joined}</div>
        """

    return f"""
    <div class="pitch-container">
        <div class="pitch-outline"></div>
        <div class="pitch-centre-line"></div>
        <div class="pitch-centre-circle"></div>
        <div class="pitch-box-top"></div>
        <div class="pitch-box-bottom"></div>
        {rows_html}
    </div>
    """


def suggestion_card_html(player_in, player_out, fixtures_list=None):
    """Render a transfer suggestion card (HTML only, no button)."""
    pos = player_in["position"]
    pos_colors = {
        "GK": ("background:#fff8e1;color:#f9a825;", "GK"),
        "DEF": ("background:#e3f2fd;color:#1565c0;", "DEF"),
        "MID": ("background:#e8f5e9;color:#2e7d32;", "MID"),
        "FWD": ("background:#fce4ec;color:#c62828;", "FWD"),
    }
    pos_style, pos_text = pos_colors.get(pos, ("background:#f0f0f0;color:#666;", pos))

    xgb_pts = player_in["xgb_predicted_pts"]
    llm_pts = player_in["llm_predicted_pts"]
    value = player_in["combined_value_score"]
    cost_delta = player_in["price"] - player_out["price"]

    xgb_delta = xgb_pts - player_out["xgb_predicted_pts"]
    llm_delta = llm_pts - player_out["llm_predicted_pts"]
    val_delta = value - player_out["combined_value_score"]

    def _delta_span(d, fmt=".1f"):
        sign = "+" if d >= 0 else ""
        cls = "delta-positive" if d > 0 else ("delta-negative" if d < 0 else "")
        style = f' class="{cls}"' if cls else ' style="color:#888;"'
        return f'<span class="s-delta"{style}>{sign}{d:{fmt}}</span>'

    cost_sign = "+" if cost_delta >= 0 else ""
    cost_cls = "delta-negative" if cost_delta > 0 else "delta-positive" if cost_delta < 0 else ""
    cost_style = f' class="{cost_cls}"' if cost_cls else ' style="color:#888;"'

    fix_html = ""
    if fixtures_list:
        badges = []
        for fix in fixtures_list:
            fdr = int(fix["fdr"]) if fix.get("fdr") else 3
            ha = "H" if fix["home_away"] == "H" else "A"
            badges.append(f'<span class="fdr-badge fdr-{fdr}">{fix["opponent"]} ({ha})</span>')
        fix_html = f'<div class="sg-fixtures">{"".join(badges)}</div>'

    return f"""
    <div class="sg-card">
        <div class="sg-top">
            <span class="sg-name">{player_in['player_name']}</span>
            <span class="sg-pos-badge" style="{pos_style}">{pos_text}</span>
            <span class="sg-team-price">{player_in['team_name']} &middot; {player_in['price']:.1f}m</span>
        </div>
        <div class="sg-stats-row">
            <div class="sg-stat">
                <div class="s-label">XGB Pts</div>
                <div class="s-val">{xgb_pts:.1f}</div>
                {_delta_span(xgb_delta)}
            </div>
            <div class="sg-stat">
                <div class="s-label">LLM Pts</div>
                <div class="s-val">{llm_pts:.1f}</div>
                {_delta_span(llm_delta)}
            </div>
            <div class="sg-stat">
                <div class="s-label">Value</div>
                <div class="s-val">{value:.2f}</div>
                {_delta_span(val_delta, ".2f")}
            </div>
            <div class="sg-stat">
                <div class="s-label">Cost</div>
                <div class="s-val">{cost_sign}{cost_delta:.1f}m</div>
            </div>
            {fix_html}
        </div>
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


def metric_card_compact_html(label, value):
    """Render a small metric card for the My Team header row."""
    return f"""
    <div class="mc-item">
        <div class="mc-label">{label}</div>
        <div class="mc-value">{value}</div>
    </div>
    """


def delta_html(value, suffix="pts"):
    """Render a positive/negative delta value."""
    if value > 0:
        return f'<span class="delta-positive">+{value:.1f} {suffix}</span>'
    elif value < 0:
        return f'<span class="delta-negative">{value:.1f} {suffix}</span>'
    return f'<span style="color:#888;">0.0 {suffix}</span>'
