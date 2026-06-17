from __future__ import annotations

SPACING = {
    "xs": "0.35rem",
    "sm": "0.55rem",
    "md": "0.85rem",
    "lg": "1.15rem",
    "xl": "1.65rem",
    "2xl": "2.25rem",
}

RADIUS = {
    "sm": "6px",
    "md": "8px",
    "pill": "999px",
}

TYPOGRAPHY = {
    "eyebrow": "0.74rem",
    "body": "0.95rem",
    "large": "1.08rem",
    "section": "1.65rem",
    "metric": "1.5rem",
    "hero": "clamp(2.45rem, 7vw, 5.1rem)",
}

STATUS_COLORS = {
    "green": {"text": "#9ff3c8", "bg": "rgba(16,185,129,.16)", "border": "rgba(110,231,183,.42)"},
    "yellow": {"text": "#fde68a", "bg": "rgba(245,158,11,.18)", "border": "rgba(251,191,36,.45)"},
    "orange": {"text": "#fdba74", "bg": "rgba(249,115,22,.18)", "border": "rgba(251,146,60,.48)"},
    "red": {"text": "#fca5a5", "bg": "rgba(239,68,68,.17)", "border": "rgba(248,113,113,.48)"},
    "blue": {"text": "#93c5fd", "bg": "rgba(59,130,246,.17)", "border": "rgba(125,211,252,.42)"},
    "grey": {"text": "#cbd5e1", "bg": "rgba(148,163,184,.13)", "border": "rgba(203,213,225,.32)"},
}

CARD_STYLE = {
    "background": "rgba(15, 28, 44, 0.92)",
    "border": "rgba(148, 163, 184, 0.18)",
    "shadow": "0 18px 42px rgba(0, 0, 0, 0.24)",
    "muted": "#a9b8c9",
    "text": "#f4f8fb",
}


def _status_css() -> str:
    rules = []
    for name, colors in STATUS_COLORS.items():
        rules.append(
            ".ep-status-{name} {{color:{text}; background:{bg}; border-color:{border};}}".format(
                name=name,
                text=colors["text"],
                bg=colors["bg"],
                border=colors["border"],
            )
        )
    return "\n".join(rules)


def build_theme_css() -> str:
    """Build one Streamlit-safe CSS block from the shared design tokens."""
    return f"""
    <style>
    :root {{
      --ep-space-xs: {SPACING["xs"]};
      --ep-space-sm: {SPACING["sm"]};
      --ep-space-md: {SPACING["md"]};
      --ep-space-lg: {SPACING["lg"]};
      --ep-space-xl: {SPACING["xl"]};
      --ep-radius-sm: {RADIUS["sm"]};
      --ep-radius-md: {RADIUS["md"]};
      --ep-radius-pill: {RADIUS["pill"]};
      --ep-card-bg: {CARD_STYLE["background"]};
      --ep-card-border: {CARD_STYLE["border"]};
      --ep-card-shadow: {CARD_STYLE["shadow"]};
      --ep-text: {CARD_STYLE["text"]};
      --ep-muted: {CARD_STYLE["muted"]};
      --ep-green: {STATUS_COLORS["green"]["text"]};
      --ep-blue: {STATUS_COLORS["blue"]["text"]};
      --ep-yellow: {STATUS_COLORS["yellow"]["text"]};
      --ep-red: {STATUS_COLORS["red"]["text"]};
    }}
    .stApp {{
      background:
        radial-gradient(circle at 82% -12%, rgba(20, 184, 166, .18) 0, rgba(20, 184, 166, 0) 30%),
        radial-gradient(circle at 8% 18%, rgba(59, 130, 246, .12) 0, rgba(59, 130, 246, 0) 26%),
        linear-gradient(180deg, #07111d 0%, #0b1624 45%, #08111d 100%);
      color: var(--ep-text);
    }}
    section[data-testid="stSidebar"] {{
      background: #07111d;
      border-right: 1px solid var(--ep-card-border);
    }}
    section[data-testid="stSidebar"] * {{
      color: #dbeafe !important;
    }}
    section[data-testid="stSidebar"] a {{
      color: #c6d3e1 !important;
      opacity: 1 !important;
      font-weight: 650;
      border-radius: var(--ep-radius-md);
      margin: .12rem .35rem;
    }}
    section[data-testid="stSidebar"] a:hover {{
      background: rgba(20, 184, 166, .13) !important;
      color: #8df5e4 !important;
    }}
    section[data-testid="stSidebar"] [aria-current="page"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
      background: rgba(20, 184, 166, .18) !important;
      color: #adfff0 !important;
      font-weight: 800;
    }}
    header[data-testid="stHeader"] {{
      background: rgba(7, 17, 29, 0.92);
      color: var(--ep-text);
    }}
    .block-container {{
      padding-top: 1.35rem;
      max-width: 1160px;
    }}
    h1, h2, h3, h4, p, label, span, div {{
      letter-spacing: 0;
    }}
    h1, h2, h3, h4 {{
      color: var(--ep-text);
    }}
    p, label, span {{
      color: inherit;
    }}
    div[data-testid="stPlotlyChart"] {{
      background: rgba(15, 28, 44, .72);
      border: 1px solid var(--ep-card-border);
      border-radius: var(--ep-radius-md);
      box-shadow: var(--ep-card-shadow);
      overflow: hidden;
    }}
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    div[data-testid="stJson"] {{
      background: rgba(15, 28, 44, .72);
      border: 1px solid var(--ep-card-border);
      border-radius: var(--ep-radius-md);
      overflow: hidden;
    }}
    div[data-testid="stLinkButton"] a, div[data-testid="stPageLink"] a {{
      border-radius: var(--ep-radius-md);
    }}
    div[data-testid="stPageLink"] a {{
      color: #cfe9ff;
    }}
    div[data-testid="stPageLink"] a:hover {{
      color: #8df5e4;
      border-color: rgba(141, 245, 228, .45);
    }}
    [data-testid="stMetric"] {{
      background: var(--ep-card-bg);
      border: 1px solid var(--ep-card-border);
      border-radius: var(--ep-radius-md);
      padding: var(--ep-space-lg);
      box-shadow: var(--ep-card-shadow);
    }}
    [data-testid="stMetricValue"] {{
      color: var(--ep-text);
    }}
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div,
    [data-testid="stDateInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSlider"] {{
      color: var(--ep-text);
    }}
    [data-testid="stExpander"] {{
      background: rgba(15, 28, 44, .62);
      border: 1px solid var(--ep-card-border);
      border-radius: var(--ep-radius-md);
    }}
    .ep-eyebrow {{
      color: var(--ep-green);
      text-transform: uppercase;
      font-size: {TYPOGRAPHY["eyebrow"]};
      font-weight: 800;
      margin-bottom: var(--ep-space-xs);
    }}
    .ep-hero {{
      color: var(--ep-text);
      font-size: clamp(2.75rem, 5.6vw, 4.45rem);
      font-weight: 850;
      line-height: 0.98;
      margin: .25rem 0 .75rem;
    }}
    .ep-subtitle {{
      color: var(--ep-muted);
      font-size: 1.12rem;
      line-height: 1.45;
      max-width: 760px;
      margin-bottom: var(--ep-space-lg);
    }}
    .ep-section-kicker {{
      color: var(--ep-blue);
      font-size: {TYPOGRAPHY["eyebrow"]};
      font-weight: 800;
      text-transform: uppercase;
      margin: 1.45rem 0 .2rem;
    }}
    .ep-section-title {{
      color: var(--ep-text);
      font-size: {TYPOGRAPHY["section"]};
      font-weight: 800;
      line-height: 1.16;
      margin: 0 0 var(--ep-space-xs);
    }}
    .ep-section-copy {{
      color: var(--ep-muted);
      font-size: {TYPOGRAPHY["body"]};
      line-height: 1.5;
      margin: 0 0 var(--ep-space-lg);
      max-width: 780px;
    }}
    .ep-card, .ep-metric-card, .ep-driver-card, .ep-explanation-card, .ep-horizon-card {{
      background: var(--ep-card-bg);
      border: 1px solid var(--ep-card-border);
      border-radius: var(--ep-radius-md);
      box-shadow: var(--ep-card-shadow);
      padding: var(--ep-space-lg);
    }}
    .ep-metric-card {{
      height: 178px;
      overflow: hidden;
    }}
    .ep-driver-card {{
      min-height: 154px;
    }}
    .ep-horizon-card {{
      min-height: 150px;
      border-left: 4px solid var(--ep-blue);
    }}
    .ep-horizon-card.ep-border-green {{border-left-color: var(--ep-green);}}
    .ep-horizon-card.ep-border-yellow, .ep-horizon-card.ep-border-orange {{border-left-color: var(--ep-yellow);}}
    .ep-horizon-card.ep-border-red {{border-left-color: var(--ep-red);}}
    .ep-horizon-card.ep-border-grey {{border-left-color: #94a3b8;}}
    .ep-icon {{
      width: 38px;
      height: 38px;
      border-radius: var(--ep-radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(20, 184, 166, .13);
      color: #8df5e4;
      border: 1px solid rgba(141, 245, 228, .28);
      font-size: 1.15rem;
      font-weight: 800;
      margin-bottom: var(--ep-space-md);
    }}
    .ep-label {{
      color: var(--ep-muted);
      font-size: 0.77rem;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .ep-value {{
      color: var(--ep-text);
      font-size: {TYPOGRAPHY["metric"]};
      font-weight: 850;
      line-height: 1.14;
      margin: .25rem 0 .35rem;
      overflow-wrap: anywhere;
    }}
    .ep-horizon-card .ep-value {{
      font-size: 1.28rem;
      line-height: 1.22;
    }}
    .ep-title {{
      color: var(--ep-text);
      font-size: {TYPOGRAPHY["large"]};
      font-weight: 800;
      line-height: 1.25;
      margin: .2rem 0 .4rem;
    }}
    .ep-detail {{
      color: var(--ep-muted);
      font-size: {TYPOGRAPHY["body"]};
      line-height: 1.45;
    }}
    .ep-metric-card .ep-detail {{
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .ep-horizon-card .ep-detail {{
      font-size: .91rem;
    }}
    .ep-status {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border: 1px solid;
      border-radius: var(--ep-radius-pill);
      padding: .24rem .62rem;
      font-size: .78rem;
      font-weight: 800;
      line-height: 1.2;
      margin: .1rem .35rem .35rem 0;
    }}
    {_status_css()}
    .ep-box {{
      border: 1px solid;
      border-radius: var(--ep-radius-md);
      padding: var(--ep-space-md) var(--ep-space-lg);
      margin: var(--ep-space-sm) 0 var(--ep-space-lg);
    }}
    .ep-box-title {{
      color: var(--ep-text);
      font-weight: 800;
      margin-bottom: .2rem;
    }}
    .ep-box-info {{
      background: rgba(14, 116, 144, .16);
      border-color: rgba(125, 211, 252, .35);
    }}
    .ep-box-warning {{
      background: rgba(180, 83, 9, .16);
      border-color: rgba(251, 191, 36, .38);
    }}
    .ep-box-body {{
      color: var(--ep-muted);
      font-size: {TYPOGRAPHY["body"]};
      line-height: 1.45;
    }}
    .ep-explanation-card {{
      border-left: 4px solid var(--ep-green);
    }}
    .ep-card-row {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--ep-space-sm);
      align-items: center;
    }}
    @media (max-width: 760px) {{
      .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.25rem;
      }}
      .ep-hero {{
        font-size: 2.45rem;
      }}
      .ep-section-title {{
        font-size: 1.35rem;
      }}
      .ep-metric-card, .ep-driver-card, .ep-horizon-card {{
        min-height: auto;
      }}
    }}
    </style>
    """
