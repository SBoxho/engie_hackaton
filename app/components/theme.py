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
      padding-top: 2.05rem;
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
      max-width: 100%;
      overflow-wrap: anywhere;
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
    .ep-page-brand {{
      color: var(--ep-green);
      font-size: 1.08rem;
      font-weight: 900;
      line-height: 1.15;
      margin: .2rem 0 .5rem;
    }}
    .ep-page-title {{
      color: var(--ep-text);
      font-size: clamp(1.95rem, 4vw, 2.55rem);
      font-weight: 900;
      line-height: 1.05;
      margin: 0 0 var(--ep-space-sm);
      overflow-wrap: anywhere;
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
      box-sizing: border-box;
      min-width: 0;
      max-width: 100%;
    }}
    .ep-metric-card {{
      min-height: 178px;
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
      display: inline-flex;
      width: fit-content;
      min-width: 38px;
      min-height: 38px;
      max-width: 100%;
      box-sizing: border-box;
      border-radius: var(--ep-radius-sm);
      align-items: center;
      justify-content: center;
      background: rgba(20, 184, 166, .13);
      color: #8df5e4;
      border: 1px solid rgba(141, 245, 228, .28);
      font-size: 1.15rem;
      font-weight: 800;
      line-height: 1.08;
      text-align: center;
      overflow-wrap: anywhere;
      white-space: normal;
      padding: .34rem .42rem;
      margin-bottom: var(--ep-space-md);
    }}
    .ep-label {{
      color: var(--ep-muted);
      font-size: 0.77rem;
      font-weight: 800;
      line-height: 1.18;
      max-width: 100%;
      overflow-wrap: anywhere;
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
      max-width: 100%;
      overflow-wrap: anywhere;
    }}
    .ep-detail {{
      color: var(--ep-muted);
      font-size: {TYPOGRAPHY["body"]};
      line-height: 1.45;
      max-width: 100%;
      overflow-wrap: anywhere;
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
      display: inline-block;
      width: fit-content;
      height: auto !important;
      flex: none !important;
      align-self: flex-start !important;
      min-width: 0;
      min-height: 0 !important;
      max-width: 100%;
      max-height: 2rem;
      box-sizing: border-box;
      border: 1px solid;
      border-radius: var(--ep-radius-sm);
      padding: .24rem .62rem;
      font-size: .78rem;
      font-weight: 800;
      line-height: 1.2;
      text-align: center;
      overflow-wrap: anywhere;
      white-space: normal;
      margin: .1rem .35rem .35rem 0;
    }}
    {_status_css()}
    .ep-provenance-row, .ep-context-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: .35rem;
      align-items: center;
    }}
    .ep-provenance {{
      display: inline-block;
      gap: .32rem;
      width: fit-content;
      height: auto !important;
      flex: none !important;
      align-self: flex-start !important;
      min-width: 0;
      min-height: 0 !important;
      max-width: 100%;
      max-height: 2rem;
      box-sizing: border-box;
      border: 1px solid rgba(203, 213, 225, .3);
      border-radius: var(--ep-radius-sm);
      padding: .22rem .55rem;
      background: rgba(15, 28, 44, .66);
      color: #e0f2fe;
      font-size: .74rem;
      font-weight: 850;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }}
    .ep-provenance-key {{
      min-width: 0;
      max-width: 100%;
      overflow-wrap: anywhere;
    }}
    .ep-provenance-official {{border-color: rgba(125, 211, 252, .45);}}
    .ep-provenance-observed {{border-color: rgba(110, 231, 183, .42);}}
    .ep-provenance-model {{border-color: rgba(251, 191, 36, .42);}}
    .ep-provenance-modelled {{border-color: rgba(251, 191, 36, .42);}}
    .ep-provenance-scenario {{border-color: rgba(147, 197, 253, .46);}}
    .ep-provenance-fallback {{border-color: rgba(251, 146, 60, .5);}}
    .ep-provenance-replay {{border-color: rgba(203, 213, 225, .38);}}
    .ep-provenance-unavailable {{border-color: rgba(203, 213, 225, .38);}}
    .ep-context-bar {{
      display: grid;
      grid-template-columns: minmax(170px, .9fr) minmax(140px, .9fr) minmax(210px, 1.2fr) minmax(210px, 1.2fr);
      gap: var(--ep-space-md);
      align-items: stretch;
      margin: .25rem 0 var(--ep-space-lg);
      padding: var(--ep-space-md);
      background: rgba(7, 17, 29, .78);
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: var(--ep-radius-md);
      box-shadow: var(--ep-card-shadow);
    }}
    .ep-context-main, .ep-context-item {{
      min-width: 0;
      padding: .55rem .65rem;
      border: 1px solid rgba(148, 163, 184, .16);
      border-radius: var(--ep-radius-sm);
      background: rgba(15, 28, 44, .58);
    }}
    .ep-context-mode {{
      color: #f8fafc;
      font-size: 1.05rem;
      font-weight: 900;
      line-height: 1.18;
    }}
    .ep-context-demo {{
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      box-sizing: border-box;
      margin-top: .35rem;
      padding: .2rem .5rem;
      border: 1px solid rgba(251, 191, 36, .46);
      border-radius: var(--ep-radius-pill);
      color: #fde68a;
      background: rgba(180, 83, 9, .18);
      font-size: .74rem;
      font-weight: 900;
      line-height: 1.2;
      overflow-wrap: anywhere;
      white-space: normal;
      text-transform: uppercase;
    }}
    .ep-context-item span {{
      display: block;
      color: var(--ep-muted);
      font-size: .72rem;
      font-weight: 850;
      text-transform: uppercase;
      margin-bottom: .2rem;
    }}
    .ep-context-item strong {{
      display: block;
      color: var(--ep-text);
      font-size: .92rem;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .ep-context-item small {{
      display: block;
      color: #b6c7d8;
      margin-top: .12rem;
      font-size: .78rem;
    }}
    .ep-context-badges {{
      grid-column: 1 / -1;
      padding: .1rem .2rem;
    }}
    .ep-trust-stack {{
      display: grid;
      gap: var(--ep-space-sm);
      margin: .2rem 0 var(--ep-space-lg);
    }}
    .ep-trust-state {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: var(--ep-space-md);
      align-items: start;
      padding: var(--ep-space-md) var(--ep-space-lg);
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: var(--ep-radius-md);
      background: rgba(15, 28, 44, .7);
    }}
    .ep-trust-label {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: fit-content;
      min-width: min(76px, 100%);
      max-width: 100%;
      box-sizing: border-box;
      text-align: center;
      padding: .24rem .5rem;
      border: 1px solid rgba(203, 213, 225, .32);
      border-radius: var(--ep-radius-pill);
      color: #e0f2fe;
      font-size: .72rem;
      font-weight: 900;
      line-height: 1.2;
      overflow-wrap: anywhere;
      white-space: normal;
      text-transform: uppercase;
    }}
    .ep-trust-title {{
      color: var(--ep-text);
      font-weight: 850;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .ep-trust-body {{
      color: var(--ep-muted);
      font-size: .92rem;
      line-height: 1.42;
      margin-top: .1rem;
    }}
    .ep-trust-stale .ep-trust-label,
    .ep-trust-fallback .ep-trust-label,
    .ep-trust-partial .ep-trust-label {{
      color: #fde68a;
      border-color: rgba(251, 191, 36, .42);
      background: rgba(180, 83, 9, .16);
    }}
    .ep-trust-error .ep-trust-label {{
      color: #fca5a5;
      border-color: rgba(248, 113, 113, .48);
      background: rgba(239, 68, 68, .17);
    }}
    .ep-term {{
      position: relative;
      display: inline-flex;
      align-items: center;
      border-bottom: 1px dotted rgba(186, 230, 253, .7);
      outline: none;
    }}
    .ep-term:focus-visible {{
      box-shadow: 0 0 0 3px rgba(125, 211, 252, .28);
      border-radius: var(--ep-radius-sm);
    }}
    .ep-term abbr {{
      text-decoration: none;
      cursor: help;
    }}
    .ep-term-popover {{
      position: absolute;
      left: 0;
      bottom: calc(100% + .45rem);
      z-index: 5;
      width: min(280px, 80vw);
      opacity: 0;
      pointer-events: none;
      background: #07111d;
      border: 1px solid rgba(125, 211, 252, .34);
      border-radius: var(--ep-radius-md);
      color: #e2e8f0;
      padding: .65rem .75rem;
      box-shadow: var(--ep-card-shadow);
      font-size: .86rem;
      line-height: 1.35;
    }}
    .ep-term:hover .ep-term-popover,
    .ep-term:focus .ep-term-popover {{
      opacity: 1;
    }}
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
    .ep-explanation-card-link {{
      position: relative;
      cursor: pointer;
      transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
    }}
    .ep-explanation-card-link:hover {{
      transform: translateY(-2px);
      box-shadow: 0 14px 28px rgba(0, 0, 0, .32);
      border-color: var(--ep-green);
    }}
    .ep-card-stretched-link {{
      position: absolute;
      inset: 0;
      z-index: 1;
      text-decoration: none;
      border-radius: inherit;
    }}
    .ep-card-stretched-link:focus-visible {{
      outline: 2px solid var(--ep-green);
      outline-offset: 2px;
    }}
    .ep-card-row {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--ep-space-sm);
      align-items: center;
      min-width: 0;
      max-width: 100%;
    }}
    .ep-page-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: var(--ep-space-lg);
      margin: .25rem 0 var(--ep-space-md);
    }}
    .ep-ribbon-cell {{
      min-height: 92px;
      background: rgba(15, 28, 44, .78);
      border: 1px solid rgba(148, 163, 184, .18);
      border-radius: var(--ep-radius-md);
      padding: var(--ep-space-md);
    }}
    .ep-ribbon-cell .ep-value {{
      font-size: 1rem;
      line-height: 1.18;
    }}
    .ep-now-hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(260px, .75fr);
      gap: var(--ep-space-lg);
      align-items: stretch;
      min-width: 0;
      box-sizing: border-box;
      margin: .35rem 0 var(--ep-space-lg);
      padding: clamp(1rem, 2.5vw, 1.65rem);
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: var(--ep-radius-md);
      background: linear-gradient(135deg, rgba(15, 28, 44, .96), rgba(8, 47, 73, .74));
      box-shadow: var(--ep-card-shadow);
    }}
    .ep-now-hero h1 {{
      color: var(--ep-text);
      font-size: clamp(2.2rem, 4.8vw, 4rem);
      line-height: 1;
      margin: .1rem 0 .65rem;
      font-weight: 900;
    }}
    .ep-now-hero p {{
      color: #d7e4ef;
      font-size: 1.08rem;
      line-height: 1.45;
      max-width: 780px;
      overflow-wrap: anywhere;
      margin: 0;
    }}
    .ep-now-hero-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: var(--ep-space-md);
    }}
    .ep-now-hero-grid div {{
      min-width: 0;
      padding: var(--ep-space-md);
      border: 1px solid rgba(148, 163, 184, .18);
      border-radius: var(--ep-radius-sm);
      background: rgba(7, 17, 29, .46);
    }}
    .ep-now-hero-grid span {{
      display: block;
      color: var(--ep-muted);
      font-size: .72rem;
      font-weight: 850;
      text-transform: uppercase;
      margin-bottom: .22rem;
    }}
    .ep-now-hero-grid strong {{
      display: block;
      color: var(--ep-text);
      font-size: 1rem;
      line-height: 1.28;
      overflow-wrap: anywhere;
    }}
    .ep-now-hero-grid small {{
      color: #b6c7d8;
      display: block;
      margin-top: .18rem;
    }}
    .ep-status-row-wrap {{
      display: grid;
      gap: var(--ep-space-md);
      margin: .35rem 0 var(--ep-space-lg);
    }}
    .ep-source-status-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--ep-space-md);
      align-items: start;
      padding: var(--ep-space-lg);
      border: 1px solid rgba(125, 211, 252, .24);
      border-radius: var(--ep-radius-md);
      background: rgba(15, 28, 44, .78);
      box-shadow: var(--ep-card-shadow);
    }}
    .ep-source-status-row > div {{
      min-width: 0;
    }}
    .ep-source-status-modelled {{
      border-color: rgba(251, 191, 36, .28);
    }}
    .ep-status-actions {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      align-items: flex-start;
      align-content: flex-start;
      gap: .25rem;
      min-width: 0;
      max-width: 100%;
    }}
    .ep-status-actions .ep-status,
    .ep-status-actions .ep-provenance,
    .ep-card-row .ep-status,
    .ep-card-row .ep-provenance {{
      display: inline-block !important;
      width: fit-content !important;
      height: auto !important;
      min-height: 0 !important;
      max-height: 2rem !important;
      align-self: flex-start !important;
      flex: none !important;
      place-self: start;
      border-radius: var(--ep-radius-sm) !important;
    }}
    .ep-source-status-row .ep-status,
    .ep-source-status-row .ep-provenance {{
      display: inline-block !important;
      width: fit-content !important;
      height: auto !important;
      min-height: 0 !important;
      max-height: 2rem !important;
      align-self: flex-start !important;
      flex: none !important;
      border-radius: var(--ep-radius-sm) !important;
    }}
    .ep-status-source {{
      grid-column: 1 / -1;
      color: var(--ep-muted);
      font-size: .88rem;
      line-height: 1.35;
      padding-top: .15rem;
      border-top: 1px solid rgba(148, 163, 184, .14);
    }}
    .ep-next12 {{
      margin: .25rem 0 var(--ep-space-lg);
    }}
    .ep-next12-date {{
      color: #bae6fd;
      font-size: .76rem;
      font-weight: 900;
      text-transform: uppercase;
      margin: .7rem 0 .35rem;
    }}
    .ep-next12 div[data-testid="stButton"] button {{
      min-height: 82px;
      padding: .42rem .35rem;
      border-radius: var(--ep-radius-sm);
      white-space: pre-line;
      line-height: 1.18;
      font-size: .84rem;
      font-weight: 760;
      overflow-wrap: anywhere;
    }}
    .ep-story-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--ep-space-md);
      margin: .25rem 0 var(--ep-space-xl);
    }}
    .ep-story-step {{
      position: relative;
      min-height: 184px;
      background: linear-gradient(145deg, rgba(15, 28, 44, .96), rgba(11, 38, 55, .88));
      border: 1px solid rgba(148, 163, 184, .24);
      border-radius: var(--ep-radius-md);
      box-shadow: var(--ep-card-shadow);
      padding: var(--ep-space-lg);
    }}
    .ep-story-number {{
      display: inline-flex;
      width: fit-content;
      min-width: 34px;
      min-height: 34px;
      max-width: 100%;
      box-sizing: border-box;
      align-items: center;
      justify-content: center;
      border-radius: var(--ep-radius-sm);
      margin-bottom: var(--ep-space-md);
      background: #e0f2fe;
      color: #082f49;
      font-weight: 900;
      line-height: 1.08;
      overflow-wrap: anywhere;
      padding: .28rem .42rem;
      border: 1px solid rgba(224, 242, 254, .72);
    }}
    .ep-source-wrap {{
      margin: .2rem 0 var(--ep-space-lg);
    }}
    .ep-source-row {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--ep-space-sm);
      margin-top: var(--ep-space-xs);
    }}
    .ep-source-badge {{
      display: inline-flex;
      align-items: center;
      flex-wrap: wrap;
      gap: .42rem;
      width: fit-content;
      min-width: 0;
      max-width: 100%;
      box-sizing: border-box;
      border: 1px solid rgba(203, 213, 225, .34);
      border-radius: var(--ep-radius-pill);
      padding: .42rem .72rem;
      background: rgba(15, 28, 44, .72);
      color: #f8fafc;
      font-size: .84rem;
      font-weight: 800;
      line-height: 1.1;
      overflow-wrap: anywhere;
      white-space: normal;
    }}
    .ep-source-badge small {{
      color: #cbd5e1;
      font-size: .74rem;
      font-weight: 650;
      margin-left: .1rem;
      min-width: 0;
      max-width: 100%;
      overflow-wrap: anywhere;
    }}
    .ep-source-dot {{
      width: .56rem;
      height: .56rem;
      border-radius: var(--ep-radius-pill);
      background: #2dd4bf;
      box-shadow: 0 0 0 3px rgba(45, 212, 191, .14);
      flex: 0 0 auto;
    }}
    .ep-viz-note {{
      display: flex;
      justify-content: space-between;
      gap: var(--ep-space-md);
      align-items: flex-start;
      padding: .78rem .95rem;
      margin: .35rem 0 .65rem;
      background: rgba(15, 28, 44, .72);
      border: 1px solid rgba(148, 163, 184, .2);
      border-radius: var(--ep-radius-md);
    }}
    .ep-viz-note > div {{
      min-width: 0;
    }}
    .ep-viz-title {{
      color: var(--ep-text);
      font-weight: 850;
      line-height: 1.25;
    }}
    .ep-viz-detail {{
      color: var(--ep-muted);
      font-size: .92rem;
      line-height: 1.42;
      margin-top: .12rem;
    }}
    .ep-viz-source {{
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      box-sizing: border-box;
      white-space: normal;
      color: #bae6fd;
      border: 1px solid rgba(125, 211, 252, .32);
      background: rgba(14, 116, 144, .16);
      border-radius: var(--ep-radius-pill);
      padding: .24rem .55rem;
      font-size: .75rem;
      font-weight: 850;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }}
    .ep-chain {{
      display: grid;
      grid-template-columns: minmax(130px, 1fr) auto minmax(130px, 1fr) auto minmax(130px, 1fr) auto minmax(130px, 1fr) auto minmax(130px, 1fr);
      gap: var(--ep-space-sm);
      align-items: stretch;
      margin: .55rem 0 var(--ep-space-xl);
    }}
    .ep-chain-node {{
      min-width: 0;
      min-height: 150px;
      padding: var(--ep-space-md);
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: var(--ep-radius-md);
      background: linear-gradient(145deg, rgba(15, 28, 44, .94), rgba(8, 47, 73, .58));
      box-shadow: var(--ep-card-shadow);
    }}
    .ep-chain-index {{
      display: inline-flex;
      width: fit-content;
      min-width: 30px;
      min-height: 30px;
      max-width: 100%;
      box-sizing: border-box;
      align-items: center;
      justify-content: center;
      border-radius: var(--ep-radius-sm);
      margin-bottom: var(--ep-space-sm);
      background: rgba(224, 242, 254, .92);
      color: #082f49;
      font-weight: 900;
      line-height: 1.08;
      overflow-wrap: anywhere;
      padding: .24rem .36rem;
    }}
    .ep-chain-node .ep-title {{
      font-size: .98rem;
      line-height: 1.18;
      margin-bottom: .25rem;
    }}
    .ep-chain-node .ep-detail {{
      font-size: .86rem;
      line-height: 1.34;
    }}
    .ep-chain-arrow {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: #93c5fd;
      font-size: 1.25rem;
      font-weight: 900;
      padding: 0 .05rem;
    }}
    .ep-why-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: var(--ep-space-md);
      margin-bottom: var(--ep-space-lg);
    }}
    .ep-why-item {{
      background: rgba(15, 28, 44, .78);
      border: 1px solid rgba(148, 163, 184, .2);
      border-radius: var(--ep-radius-md);
      padding: var(--ep-space-lg);
      min-height: 142px;
    }}
    .ep-how-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--ep-space-md);
      margin: var(--ep-space-md) 0;
    }}
    .ep-how-item {{
      background: rgba(7, 17, 29, .42);
      border: 1px solid rgba(148, 163, 184, .18);
      border-radius: var(--ep-radius-md);
      padding: var(--ep-space-md);
    }}
    .ep-footer {{
      display: flex;
      justify-content: space-between;
      gap: var(--ep-space-lg);
      align-items: center;
      margin: 2rem 0 .5rem;
      padding: var(--ep-space-xl);
      border-top: 1px solid rgba(148, 163, 184, .24);
      background: rgba(7, 17, 29, .5);
    }}
    .ep-footer-title {{
      color: var(--ep-text);
      font-size: 1.28rem;
      font-weight: 850;
      margin: .2rem 0 .28rem;
      overflow-wrap: anywhere;
    }}
    .ep-footer-team {{
      color: #e0f2fe;
      font-weight: 850;
      text-align: right;
    }}
    @media (max-width: 760px) {{
      html, body, .stApp {{
        overflow-x: hidden;
      }}
      div[data-testid="stAppViewContainer"],
      div[data-testid="stMain"],
      main,
      .block-container {{
        min-width: 0 !important;
        max-width: 100vw !important;
        width: 100% !important;
        box-sizing: border-box;
      }}
      div[data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
      }}
      .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.75rem;
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
      .ep-story-grid, .ep-why-grid, .ep-how-grid, .ep-chain {{
        grid-template-columns: 1fr;
      }}
      .ep-chain-node {{
        min-height: auto;
      }}
      .ep-chain-arrow {{
        display: none;
      }}
      .ep-now-hero, .ep-source-status-row {{
        grid-template-columns: 1fr;
        width: 100%;
        max-width: 100%;
      }}
      .ep-context-bar,
      .ep-trust-state,
      .ep-status-row-wrap,
      .ep-viz-note,
      .ep-card,
      .ep-metric-card,
      .ep-driver-card,
      .ep-explanation-card,
      .ep-horizon-card {{
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
      }}
      .ep-section-copy,
      .ep-detail,
      .ep-trust-body,
      .ep-source-status-row {{
        overflow-wrap: anywhere;
      }}
      .ep-status-actions {{
        justify-content: flex-start;
      }}
      .ep-story-step {{
        min-height: auto;
      }}
      .ep-viz-note, .ep-footer {{
        flex-direction: column;
        align-items: stretch;
      }}
      .ep-context-bar {{
        grid-template-columns: 1fr;
      }}
      .ep-trust-state {{
        grid-template-columns: 1fr;
      }}
      .ep-viz-source {{
        width: fit-content;
      }}
      .ep-footer-team {{
        text-align: left;
      }}
    }}
    @media (min-width: 761px) and (max-width: 1120px) {{
      .ep-context-bar {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    </style>
    """
