
COLORS = {
    "background": "#F5F7FA",
    "card": "#FFFFFF",
    "primary": "#1E3A5F",
    "primary_light": "#294B75",
    "secondary": "#2E8B57",
    "accent": "#D9534F",
    "text": "#1F2937",
    "muted": "#6B7280",
    "border": "#E5E7EB",
    "tab": "#EEF2F7",
    "hover": "#F9FAFB",
    "shadow": "rgba(0,0,0,0.06)",
}

FONT_FAMILY = "'Inter', 'Segoe UI', sans-serif"

# ==========================================================
# LAYOUT
# ==========================================================

APP_STYLE = {
    "backgroundColor": COLORS["background"],
    "minHeight": "100vh",
    "fontFamily": FONT_FAMILY,
}

CONTENT_STYLE = {
    "marginLeft": "260px",
    "padding": "32px",
}

# ==========================================================
# SIDEBAR
# ==========================================================

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "240px",
    "padding": "28px 20px",
    "backgroundColor": COLORS["primary"],
    "color": "white",
    "overflowY": "auto",
    "boxShadow": "2px 0 12px rgba(0,0,0,0.08)",
    "zIndex": 999,
}

SIDEBAR_TITLE = {
    "fontSize": "1.5rem",
    "fontWeight": "700",
    "marginBottom": "12px",
}

SIDEBAR_SUBTITLE = {
    "fontSize": "0.95rem",
    "lineHeight": "1.5",
    "color": "#D1D5DB",
    "marginBottom": "32px",
}

NAV_SECTION = {
    "marginBottom": "24px",
}

NAV_TITLE = {
    "fontSize": "0.78rem",
    "fontWeight": "700",
    "letterSpacing": "1px",
    "textTransform": "uppercase",
    "color": "#93C5FD",
    "marginBottom": "10px",
}

NAV_LINK = {
    "display": "block",
    "textDecoration": "none",
    "color": "white",
    "padding": "12px 14px",
    "borderRadius": "10px",
    "marginBottom": "8px",
    "backgroundColor": "rgba(255,255,255,0.05)",
    "cursor": "pointer",
    "fontSize": "0.95rem",
    "transition": "all 0.2s ease",
}

NAV_LINK_ACTIVE = {
    **NAV_LINK,
    "backgroundColor": "rgba(255,255,255,0.14)",
    "fontWeight": "600",
}

# ==========================================================
# HERO
# ==========================================================

HERO_CONTAINER = {
    "background": "linear-gradient(135deg, #1E3A5F 0%, #294B75 100%)",
    "padding": "48px",
    "borderRadius": "24px",
    "color": "white",
    "marginBottom": "36px",
    "boxShadow": "0 8px 30px rgba(0,0,0,0.10)",
}

HERO_TITLE = {
    "fontSize": "2.5rem",
    "fontWeight": "800",
    "marginBottom": "14px",
}

HERO_TEXT = {
    "fontSize": "1.05rem",
    "lineHeight": "1.8",
    "maxWidth": "900px",
    "color": "#E5E7EB",
}

# ==========================================================
# SECTION HEADERS
# ==========================================================

SECTION_CONTAINER = {
    "marginBottom": "50px",
}

SECTION_HEADER = {
    "marginBottom": "28px",
}

SECTION_TITLE = {
    "fontSize": "1.8rem",
    "fontWeight": "700",
    "marginBottom": "8px",
    "color": COLORS["primary"],
}

SECTION_DESCRIPTION = {
    "fontSize": "1rem",
    "lineHeight": "1.7",
    "color": COLORS["muted"],
    "maxWidth": "1000px",
}



# ==========================================================
# GRAPH GRID
# ==========================================================

GRAPH_GRID = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(580px, 1fr))",
    "gap": "24px",
    "marginBottom": "50px",
}

GRAPH_CARD = {
    "backgroundColor": COLORS["card"],
    "padding": "22px",
    "borderRadius": "22px",
    "boxShadow": "0 4px 14px rgba(0,0,0,0.05)",
    "border": f"1px solid {COLORS['border']}",
    "transition": "all 0.2s ease",
}

GRAPH_TITLE = {
    "fontSize": "1.1rem",
    "fontWeight": "700",
    "marginBottom": "8px",
    "color": COLORS["text"],
}

GRAPH_DESCRIPTION = {
    "fontSize": "0.92rem",
    "lineHeight": "1.6",
    "color": COLORS["muted"],
    "marginBottom": "18px",
}

# ==========================================================
# TABLES
# ==========================================================

TABLE_CONTAINER = {
    "backgroundColor": COLORS["card"],
    "padding": "28px",
    "borderRadius": "24px",
    "boxShadow": "0 4px 18px rgba(0,0,0,0.05)",
    "border": f"1px solid {COLORS['border']}",
}

TABLE_HEADER_TITLE = {
    "fontSize": "1.35rem",
    "fontWeight": "700",
    "marginBottom": "8px",
    "color": COLORS["primary"],
}

TABLE_HEADER_DESCRIPTION = {
    "fontSize": "0.95rem",
    "color": COLORS["muted"],
    "marginBottom": "22px",
}

TABLE_STYLE = {
    "overflowX": "auto",
    "borderRadius": "18px",
    "overflow": "hidden",
}

TABLE_CELL = {
    "textAlign": "left",
    "padding": "14px",
    "fontFamily": FONT_FAMILY,
    "fontSize": "14px",
    "border": "none",
    "backgroundColor": "white",
    "color": COLORS["text"],
}

TABLE_HEADER = {
    "backgroundColor": COLORS["primary"],
    "color": "white",
    "fontWeight": "700",
    "fontSize": "14px",
    "border": "none",
    "padding": "16px",
}

TABLE_DATA = {
    "borderBottom": f"1px solid {COLORS['border']}",
}

TABLE_CONDITIONAL = [
    {
        "if": {"row_index": "odd"},
        "backgroundColor": COLORS["hover"]
    },
    {
        "if": {"state": "active"},
        "border": f"1px solid {COLORS['primary']}",
        "backgroundColor": "#EEF4FF"
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "#E0EAFF",
        "border": f"1px solid {COLORS['primary']}"
    }
]

# ==========================================================
# TABS
# ==========================================================

TAB_STYLE = {
    "padding": "14px 20px",
    "borderRadius": "12px",
    "backgroundColor": COLORS["tab"],
    "border": "none",
    "fontWeight": "600",
    "fontSize": "0.95rem",
    "marginRight": "10px",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "backgroundColor": COLORS["primary"],
    "color": "white",
}

# ==========================================================
# EMPTY STATES / ALERTS
# ==========================================================

EMPTY_STATE = {
    "padding": "50px 20px",
    "textAlign": "center",
    "color": COLORS["muted"],
    "backgroundColor": COLORS["card"],
    "borderRadius": "20px",
    "border": f"1px dashed {COLORS['border']}",
}

ALERT_CARD = {
    "padding": "18px 22px",
    "borderRadius": "16px",
    "backgroundColor": "#FEF2F2",
    "border": "1px solid #FECACA",
    "color": "#991B1B",
    "marginBottom": "20px",
}

SUCCESS_CARD = {
    "padding": "18px 22px",
    "borderRadius": "16px",
    "backgroundColor": "#ECFDF5",
    "border": "1px solid #A7F3D0",
    "color": "#065F46",
    "marginBottom": "20px",
}