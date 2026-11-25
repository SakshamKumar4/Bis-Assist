"""BIS Standards Assistant - Advanced Gemini UI with Light/Dark Mode, Sidebar Nav, Auth & Multimodal RAG."""
import os
import json
import base64
from datetime import datetime
import streamlit as st
from rag import answer, INTENT_LABELS, chunks, GEMINI_API_KEY

# Page Configuration
st.set_page_config(
    page_title="BIS Assistant ✦ Gemini",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = True

if "user_name" not in st.session_state:
    st.session_state.user_name = "Udit Taneja"

if "user_email" not in st.session_state:
    st.session_state.user_email = "udit.taneja@example.com"

if "user_initials" not in st.session_state:
    st.session_state.user_initials = "UT"

if "user_plan" not in st.session_state:
    st.session_state.user_plan = "Go"

if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"

if "active_chat_title" not in st.session_state:
    st.session_state.active_chat_title = "BIS Services Assistant"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "uploaded_image_type" not in st.session_state:
    st.session_state.uploaded_image_type = None

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-flash-lite-latest"

if "use_intent_routing" not in st.session_state:
    st.session_state.use_intent_routing = True

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

# Dynamic Light / Dark Theme Injection
is_dark = st.session_state.dark_mode

theme_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --bg-main: {'#131314' if is_dark else '#F8F9FA'};
        --bg-surface: {'#1E1F20' if is_dark else '#FFFFFF'};
        --bg-hover: {'#2B2C2F' if is_dark else '#ECEFF1'};
        --border-color: {'#2E3033' if is_dark else '#E0E2EC'};
        --text-primary: {'#E3E3E3' if is_dark else '#1F1F1F'};
        --text-secondary: {'#9AA0A6' if is_dark else '#5F6368'};
        --sidebar-bg: {'#18191A' if is_dark else '#F0F4F9'};
        --chat-assistant-bg: {'#1E1F20' if is_dark else '#FFFFFF'};
        --card-shadow: {'none' if is_dark else '0 2px 10px rgba(0,0,0,0.06)'};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Google Sans', 'Inter', -apple-system, sans-serif !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }}

    /* Top Bar Styling */
    .top-bar-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px;
        margin-bottom: 20px;
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        box-shadow: var(--card-shadow);
    }}

    .top-bar-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        font-size: 1.15rem;
    }}

    .gemini-sparkle {{
        background: linear-gradient(135deg, #4285F4 0%, #9B72CB 50%, #D96570 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
    }}

    .top-bar-model-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background-color: {'#2A2B2E' if is_dark else '#E8F0FE'};
        color: {'#8AB4F8' if is_dark else '#1A73E8'};
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    /* Title Gradient */
    .gemini-title {{
        background: linear-gradient(90deg, #4285F4 0%, #9B72CB 40%, #D96570 85%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .gemini-subtitle {{
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 20px;
    }}

    /* Sidebar Navigation Items */
    .sidebar-section-title {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary);
        margin: 16px 8px 6px 8px;
    }}

    .nav-btn {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        border-radius: 10px;
        color: var(--text-primary);
        font-size: 0.88rem;
        text-decoration: none;
        cursor: pointer;
        transition: background 0.15s ease;
        margin-bottom: 2px;
    }}

    .nav-btn:hover {{
        background-color: var(--bg-hover);
    }}

    .nav-btn.active {{
        background-color: {'#282A2D' if is_dark else '#E3EDFD'};
        font-weight: 600;
        color: {'#FFFFFF' if is_dark else '#1A73E8'};
    }}

    /* User Profile Card at Bottom of Sidebar */
    .user-profile-card {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 12px;
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        margin-top: 20px;
    }}

    .user-avatar-circle {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #A855F7 0%, #EC4899 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
    }}

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    [data-testid="stChatMessage"]:nth-child(even) {{
        background-color: var(--chat-assistant-bg) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--card-shadow) !important;
    }}

    /* Pill Badges for Intents */
    .intent-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        margin-bottom: 8px;
    }}

    .intent-hallmarking {{
        background-color: {'rgba(251, 188, 4, 0.15)' if is_dark else '#FEF7E0'};
        color: {'#FDD663' if is_dark else '#B06000'};
        border: 1px solid {'rgba(251, 188, 4, 0.35)' if is_dark else '#F9AB00'};
    }}

    .intent-standard_lookup {{
        background-color: {'rgba(66, 133, 244, 0.15)' if is_dark else '#E8F0FE'};
        color: {'#8AB4F8' if is_dark else '#1A73E8'};
        border: 1px solid {'rgba(66, 133, 244, 0.35)' if is_dark else '#4285F4'};
    }}

    .intent-certification_process {{
        background-color: {'rgba(52, 168, 83, 0.15)' if is_dark else '#E6F4EA'};
        color: {'#81C995' if is_dark else '#137333'};
        border: 1px solid {'rgba(52, 168, 83, 0.35)' if is_dark else '#34A853'};
    }}

    .intent-other {{
        background-color: {'rgba(154, 160, 166, 0.15)' if is_dark else '#F1F3F4'};
        color: {'#BDC1C6' if is_dark else '#5F6368'};
        border: 1px solid {'rgba(154, 160, 166, 0.35)' if is_dark else '#BDC1C6'};
    }}

    .source-box {{
        background: {'#18191A' if is_dark else '#F8F9FA'};
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
    }}

    .stButton>button {{
        border-radius: 20px !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease !important;
    }}

    .stButton>button:hover {{
        background-color: var(--bg-hover) !important;
        border-color: #8AB4F8 !important;
    }}

    /* ---- Chat Input Box (fixes black bar in light mode) ---- */
    [data-testid="stChatInput"] {{
        border-radius: 28px !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }}
    [data-testid="stChatInputContainer"] {{
        background-color: var(--bg-main) !important;
        border-top: 1px solid var(--border-color) !important;
    }}

    /* ---- Hide white Streamlit header bar at top ---- */
    [data-testid="stHeader"] {{
        background-color: var(--bg-main) !important;
        border-bottom: 1px solid var(--border-color) !important;
        box-shadow: none !important;
    }}

    /* ---- Hide white toolbar (Deploy button area) ---- */
    [data-testid="stToolbar"] {{
        background-color: var(--bg-main) !important;
        right: 0;
    }}
    [data-testid="stDecoration"] {{
        background: var(--bg-main) !important;
        height: 0 !important;
    }}

    /* ---- Footer / bottom bar fix ---- */
    footer {{
        background-color: var(--bg-main) !important;
        border-top: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
    }}
    footer a {{
        color: var(--text-secondary) !important;
    }}

    /* ---- Main block top padding (shrinks gap under header) ---- */
    .block-container {{
        padding-top: 1.2rem !important;
    }}

    /* ---- Text inputs / select boxes theme ---- */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    .stSelectbox [data-baseweb="select"] {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
    }}
    [data-baseweb="option"] {{
        color: var(--text-primary) !important;
    }}
    [data-baseweb="option"]:hover {{
        background-color: var(--bg-hover) !important;
    }}

    /* Metric labels */
    [data-testid="metric-container"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px;
        padding: 10px 14px;
    }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# ---- Aggressive overrides for Streamlit chrome (header, footer, bottom input) ----
st.markdown(f"""
<style>
/* === TOP HEADER BAR === */
[data-testid="stHeader"] {{
    background-color: {'#131314' if is_dark else '#F8F9FA'} !important;
    border-bottom: 1px solid {'#2E3033' if is_dark else '#E0E2EC'} !important;
    box-shadow: none !important;
    height: 3.5rem !important;
}}
[data-testid="stDecoration"] {{ height: 0 !important; display: none !important; }}
[data-testid="stToolbar"] {{
    background: {'#131314' if is_dark else '#F8F9FA'} !important;
    border: none !important;
}}
[data-testid="stToolbarActions"] button {{
    color: {'#9AA0A6' if is_dark else '#5F6368'} !important;
    background: transparent !important;
}}

/* === BOTTOM AREA & CHAT INPUT === */
[data-testid="stBottom"] {{
    background-color: {'#131314' if is_dark else '#F8F9FA'} !important;
    border-top: 1px solid {'#2E3033' if is_dark else '#E0E2EC'} !important;
    padding: 10px 0 0 0 !important;
}}
[data-testid="stBottomBlockContainer"] {{
    background-color: {'#131314' if is_dark else '#F8F9FA'} !important;
}}
.stChatFloatingInputContainer {{
    background-color: {'#131314' if is_dark else '#F8F9FA'} !important;
    border-top: 1px solid {'#2E3033' if is_dark else '#E0E2EC'} !important;
    padding: 10px 16px 12px 16px !important;
}}
[data-testid="stChatInput"] {{
    background-color: {'#1E1F20' if is_dark else '#FFFFFF'} !important;
    border: 1.5px solid {'#3C4043' if is_dark else '#DADCE0'} !important;
    border-radius: 28px !important;
    color: {'#E3E3E3' if is_dark else '#1F1F1F'} !important;
    padding: 10px 16px !important;
}}
[data-testid="stChatInput"] textarea {{
    background-color: {'#1E1F20' if is_dark else '#FFFFFF'} !important;
    color: {'#E3E3E3' if is_dark else '#1F1F1F'} !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {'#5F6368' if is_dark else '#9AA0A6'} !important;
}}

/* === FOOTER === */
footer {{
    background-color: {'#131314' if is_dark else '#F8F9FA'} !important;
    color: {'#5F6368' if is_dark else '#9AA0A6'} !important;
    border-top: 1px solid {'#2E3033' if is_dark else '#E0E2EC'} !important;
}}

/* === BLOCK CONTAINER === */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. SIDEBAR (Matching Modern Chat Layout: Pinned, Projects, Recents, Profile)
# ==============================================================================
with st.sidebar:
    # Top Action: New Chat
    if st.button("✏️  New chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.uploaded_image_bytes = None
        st.session_state.uploaded_image_type = None
        st.session_state.current_view = "chat"
        st.session_state.active_chat_title = "New Chat"
        st.rerun()

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # General Function Navigation (Matching Image 1)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.current_view = "chat"
            st.rerun()
    with col_nav2:
        if st.button("📖 Library", use_container_width=True):
            st.session_state.current_view = "library"
            st.rerun()

    col_nav3, col_nav4 = st.columns(2)
    with col_nav3:
        if st.button("🕒 Scheduled", use_container_width=True):
            st.session_state.current_view = "scheduled"
            st.rerun()
    with col_nav4:
        if st.button("🧩 Plugins", use_container_width=True):
            st.session_state.current_view = "plugins"
            st.rerun()

    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.current_view = "settings"
        st.rerun()

    st.markdown("---")

    # Pinned Section
    st.markdown("<div class='sidebar-section-title'>Pinned</div>", unsafe_allow_html=True)
    if st.button("📌 IS 302: Electrical Appliances", use_container_width=True):
        st.session_state.current_view = "chat"
        st.session_state.pending_query = "What safety requirements apply to electric appliances under IS 302?"
        st.session_state.active_chat_title = "IS 302 Electricals"
        st.rerun()

    if st.button("📌 Gold & Silver 6-Digit HUID", use_container_width=True):
        st.session_state.current_view = "chat"
        st.session_state.pending_query = "How do I verify a 6-digit HUID hallmark on gold jewellery?"
        st.session_state.active_chat_title = "HUID Hallmarking"
        st.rerun()

    if st.button("📌 BIS Scheme-I ISI Process", use_container_width=True):
        st.session_state.current_view = "chat"
        st.session_state.pending_query = "What is the procedure and documentation needed for BIS Scheme-I certification?"
        st.session_state.active_chat_title = "Scheme-I Process"
        st.rerun()

    # Projects Section
    st.markdown("<div class='sidebar-section-title'>Projects</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.83rem; color:var(--text-secondary); padding-left:10px;'>No projects</div>", unsafe_allow_html=True)

    # Recents Section
    st.markdown("<div class='sidebar-section-title'>Recents</div>", unsafe_allow_html=True)
    recents = [
        "BIS Services Assistant",
        "Electric Iron Certification",
        "Jewellery Hallmarking Standards",
        "Packaged Drinking Water IS 14543"
    ]
    for r_title in recents:
        is_active = (r_title == st.session_state.active_chat_title)
        btn_label = f"💬 {r_title}" if not is_active else f"▶ {r_title}"
        if st.button(btn_label, use_container_width=True, key=f"rec_{r_title}"):
            st.session_state.current_view = "chat"
            st.session_state.active_chat_title = r_title
            st.session_state.pending_query = f"Tell me more about {r_title}"
            st.rerun()

    st.markdown("<div style='flex-grow:1;'></div>", unsafe_allow_html=True)

    # Bottom User Profile Card (Matching Image 1)
    st.markdown("---")
    col_u1, col_u2 = st.columns([1, 3])
    with col_u1:
        st.markdown(
            f"<div class='user-avatar-circle'>{st.session_state.user_initials}</div>",
            unsafe_allow_html=True
        )
    with col_u2:
        st.markdown(
            f"<div style='line-height:1.2;'>"
            f"<div style='font-weight:600; font-size:0.9rem;'>{st.session_state.user_name}</div>"
            f"<div style='font-size:0.75rem; color:var(--text-secondary);'>{st.session_state.user_plan} · Active</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Profile action buttons
    col_prof1, col_prof2 = st.columns(2)
    with col_prof1:
        if st.button("👤 Account", use_container_width=True):
            st.session_state.current_view = "auth"
            st.rerun()
    with col_prof2:
        if st.session_state.is_authenticated:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.is_authenticated = False
                st.session_state.user_name = "Guest User"
                st.session_state.user_initials = "GU"
                st.session_state.current_view = "auth"
                st.rerun()
        else:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.current_view = "auth"
                st.rerun()

# ==============================================================================
# 2. TOP BAR (single clean HTML row - Light/Dark toggle + status chip + breadcrumb)
# ==============================================================================
has_api_key = bool(GEMINI_API_KEY.strip() if GEMINI_API_KEY else None or os.getenv("GEMINI_API_KEY"))
status_color = "#81C995" if has_api_key else "#FDD663"
status_text = "Gemini Connected" if has_api_key else "RAG Fallback Mode"
current_model = st.session_state.gemini_model
current_title = st.session_state.active_chat_title

# Render the HTML brand+status side of the top bar
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding: 8px 18px; margin-bottom: 14px;
            background: {'#1E1F20' if is_dark else '#FFFFFF'};
            border: 1px solid {'#2E3033' if is_dark else '#E0E2EC'};
            border-radius: 14px; box-shadow: {'none' if is_dark else '0 1px 6px rgba(0,0,0,0.07)'};
            ">
  <div style="display:flex; align-items:center; gap:10px; flex:1;">
    <span style="background:linear-gradient(135deg,#4285F4,#9B72CB,#D96570);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 font-size:1.4rem; font-weight:800;">✦</span>
    <span style="font-weight:700; font-size:1.05rem; color:{'#E3E3E3' if is_dark else '#1F1F1F'};">BIS Assistant</span>
    <span style="font-size:0.8rem; color:{'#9AA0A6' if is_dark else '#5F6368'}; font-weight:400;">/ {current_title}</span>
    <span style="margin-left:8px; padding:3px 10px;
                 background:{'#2A2B2E' if is_dark else '#E8F0FE'};
                 color:{'#8AB4F8' if is_dark else '#1A73E8'};
                 border-radius:16px; font-size:0.75rem; font-weight:600; white-space:nowrap;">
      <span style="color:{status_color};">●</span>&nbsp;{current_model}&nbsp;·&nbsp;{status_text}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# Render the toggle in a right-aligned container separately (Streamlit widget must be in Python)
col_tgl, col_right = st.columns([5, 1])
with col_right:
    current_mode_is_dark = st.toggle(
        "🌙" if st.session_state.dark_mode else "☀️",
        value=st.session_state.dark_mode,
        key="top_bar_theme_toggle",
        help="Toggle Light / Dark mode"
    )
    if current_mode_is_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = current_mode_is_dark
        st.rerun()

# ==============================================================================
# 3. VIEW ROUTER: AUTH (LOGIN / SIGN UP) VIEW
# ==============================================================================
if st.session_state.current_view == "auth":
    st.markdown("""
        <div style="text-align:center; max-width:550px; margin: 20px auto 30px auto;">
            <div class="gemini-sparkle" style="font-size:3rem; margin-bottom:8px;">✦</div>
            <div style="font-size:2rem; font-weight:700;">Welcome to BIS Assistant</div>
            <div style="color:var(--text-secondary); font-size:0.95rem;">
                Sign in to sync your conversation history, saved standards, and compliance audits.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_spacer_l, col_auth_card, col_spacer_r = st.columns([1, 2, 1])
    with col_auth_card:
        auth_tab_login, auth_tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

        with auth_tab_login:
            st.markdown("### Sign In to your Account")
            login_email = st.text_input("Email or Username", value=st.session_state.user_email if st.session_state.is_authenticated else "")
            login_pass = st.text_input("Password", type="password", value="password123")
            remember_me = st.checkbox("Remember this device", value=True)

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                if st.button("Sign In", type="primary", use_container_width=True):
                    name_part = login_email.split("@")[0].replace(".", " ").title() if "@" in login_email else login_email
                    st.session_state.is_authenticated = True
                    st.session_state.user_name = name_part or "Udit Taneja"
                    st.session_state.user_email = login_email
                    st.session_state.user_initials = "".join([part[0] for part in st.session_state.user_name.split()[:2]]).upper() or "UT"
                    st.session_state.current_view = "chat"
                    st.success(f"Welcome back, {st.session_state.user_name}!")
                    st.rerun()
            with col_sub2:
                if st.button("Continue as Guest", use_container_width=True):
                    st.session_state.is_authenticated = True
                    st.session_state.user_name = "Guest Explorer"
                    st.session_state.user_initials = "GE"
                    st.session_state.current_view = "chat"
                    st.rerun()

        with auth_tab_signup:
            st.markdown("### Create New BIS Account")
            new_name = st.text_input("Full Name", placeholder="e.g. Udit Taneja")
            new_email = st.text_input("Official Email Address", placeholder="name@company.com")
            new_role = st.selectbox(
                "Role / Industry Type",
                ["Manufacturer / Exporter", "Jeweller / Gold Hallmarking", "Consumer / Citizen", "Quality Assurance Auditor", "Student / Researcher"]
            )
            new_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            if st.button("Create Account", type="primary", use_container_width=True):
                if new_name and new_email and new_password:
                    if new_password == confirm_password:
                        st.session_state.is_authenticated = True
                        st.session_state.user_name = new_name
                        st.session_state.user_email = new_email
                        st.session_state.user_initials = "".join([part[0] for part in new_name.split()[:2]]).upper() or "UT"
                        st.session_state.current_view = "chat"
                        st.success(f"Account successfully created for {new_name}!")
                        st.rerun()
                    else:
                        st.error("Passwords do not match. Please re-enter.")
                else:
                    st.error("Please fill in all required fields.")

    st.stop()

# ==============================================================================
# 4. VIEW ROUTER: LIBRARY VIEW (BIS Documents & Specifications)
# ==============================================================================
elif st.session_state.current_view == "library":
    st.markdown("""
        <div class="gemini-title">
            <span>📖</span>
            <span>BIS Standards Library</span>
        </div>
        <div class="gemini-subtitle">
            Search and browse indexed Indian Standards documents, certification scheme manuals, and hallmarking rules.
        </div>
    """, unsafe_allow_html=True)

    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.metric("Total Indexed Chunks", len(chunks))
    with col_l2:
        st.metric("Covered Sectors", "Electrical · Food · Toys")
    with col_l3:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")

    st.markdown("### Repository Documents")
    for idx, c in enumerate(chunks):
        with st.expander(f"📄 {c.get('source', 'Doc')} (Page {c.get('page', 1)}) — Category: {c.get('category', 'N/A').title()}", expanded=(idx == 0)):
            st.markdown(f"**Document Type**: `{c.get('doc_type', 'general')}`")
            st.write(c.get("text", ""))

    st.markdown("---")
    st.info("💡 To add real PDFs or TXT standards, place them in `data/raw/<category>/` and re-run `python build_index.py`.")
    st.stop()

# ==============================================================================
# 5. VIEW ROUTER: SCHEDULED & COMPLIANCE VIEW
# ==============================================================================
elif st.session_state.current_view == "scheduled":
    st.markdown("""
        <div class="gemini-title">
            <span>🕒</span>
            <span>Scheduled Audits & Compliance</span>
        </div>
        <div class="gemini-subtitle">
            Track statutory BIS certification renewals, surveillance audits, and standard updates.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    | Task / Regulation | Product Category | Standard Code | Deadline | Status |
    |---|---|---|---|---|
    | **Mandatory ISI Mark Surveillance** | Electrical Appliances | IS 302-1:2008 | Nov 15, 2026 | 🟡 Pending Audit |
    | **Annual Scheme-I License Renewal** | Food Products | IS 14543:2016 | Dec 01, 2026 | 🟢 Up to Date |
    | **6-Digit HUID Quarterly Filing** | Gold Jewellery | IS 1417:2016 | Oct 31, 2026 | 🔵 In Review |
    | **Toy Safety Quality Control Order** | Children Toys | IS 9873:2020 | Jan 10, 2027 | ⚪ Scheduled |
    """)
    st.stop()

# ==============================================================================
# 6. VIEW ROUTER: PLUGINS VIEW
# ==============================================================================
elif st.session_state.current_view == "plugins":
    st.markdown("""
        <div class="gemini-title">
            <span>🧩</span>
            <span>BIS Ecosystem Plugins</span>
        </div>
        <div class="gemini-subtitle">
            Extend assistant capabilities with official Bureau of Indian Standards integrations.
        </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
            <div class="source-box">
                <div style="font-weight:700; font-size:1.05rem; color:#8AB4F8;">🛡️ Manakonline Portal Connector</div>
                <p style="font-size:0.85rem; color:var(--text-secondary); margin-top:6px;">
                    Integrate directly with Manakonline to check real-time application tracking, fee calculators, and lab testing queues.
                </p>
                <span class="top-bar-model-chip">🟢 Connected</span>
            </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown("""
            <div class="source-box">
                <div style="font-weight:700; font-size:1.05rem; color:#FDD663;">💎 BIS Care App HUID Validator</div>
                <p style="font-size:0.85rem; color:var(--text-secondary); margin-top:6px;">
                    Verify 6-digit alphanumeric HUID numbers against central hallmarking databases to authenticate gold purity & jeweller registration.
                </p>
                <span class="top-bar-model-chip">🟢 Ready</span>
            </div>
        """, unsafe_allow_html=True)
    st.stop()

# ==============================================================================
# 7. VIEW ROUTER: SETTINGS VIEW
# ==============================================================================
elif st.session_state.current_view == "settings":
    st.markdown("""
        <div class="gemini-title">
            <span>⚙️</span>
            <span>System & Model Settings</span>
        </div>
        <div class="gemini-subtitle">
            Configure Google Gemini generation parameters and vector retrieval settings.
        </div>
    """, unsafe_allow_html=True)

    st.session_state.gemini_model = st.selectbox(
        "Google Gemini Model",
        [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-pro-latest",
        ],
        index=0,
        help="gemini-flash-lite-latest is the most stable free-tier model. Others are tried automatically if one is busy."
    )

    st.session_state.use_intent_routing = st.toggle(
        "Intent-Based Routing",
        value=st.session_state.use_intent_routing,
        help="Route queries to specific document categories (standard_lookup, certification_process, hallmarking)."
    )

    st.session_state.top_k = st.slider(
        "Top Passages to Retrieve (k)",
        min_value=1,
        max_value=8,
        value=st.session_state.top_k
    )

    st.markdown("---")
    st.success("API Key is configured directly in code (`rag.py` / `.env`). No manual UI entry needed!")
    st.stop()

# ==============================================================================
# 8. VIEW ROUTER: CHAT VIEW (Main Assistant with Multimodal Support)
# ==============================================================================
# Header Section
st.markdown("""
    <div class="gemini-title">
        <span class="gemini-sparkle">✦</span>
        <span>BIS Standards Assistant</span>
    </div>
    <div class="gemini-subtitle">
        Natural language intelligence over Indian Standards, ISI certification processes, and gold/silver HUID hallmarking.
    </div>
""", unsafe_allow_html=True)

# Media Upload Expander (for inspecting photos of marks/stamps/labels)
with st.expander("📸 Attach Media (Inspect ISI Marks, Hallmark Stamps, or Product Labels)", expanded=False):
    st.markdown(
        "<span style='font-size:0.85rem; color:var(--text-secondary);'>"
        "Upload a photo of an ISI mark, 6-digit HUID hallmark, or product packaging. "
        "Gemini's multimodal vision will inspect it against Indian Standards."
        "</span>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.session_state.uploaded_image_bytes = file_bytes
        st.session_state.uploaded_image_type = uploaded_file.type
        
        col_prev, col_info = st.columns([1, 3])
        with col_prev:
            st.image(file_bytes, caption="Attached Image", width=180)
        with col_info:
            st.success(f"Attached: {uploaded_file.name} ({len(file_bytes) // 1024} KB)")
            if st.button("✕ Remove Media"):
                st.session_state.uploaded_image_bytes = None
                st.session_state.uploaded_image_type = None
                st.rerun()

    elif st.session_state.uploaded_image_bytes is not None:
        col_prev, col_info = st.columns([1, 3])
        with col_prev:
            st.image(st.session_state.uploaded_image_bytes, caption="Active Attachment", width=180)
        with col_info:
            st.info("Image currently active for next query.")
            if st.button("✕ Remove Media"):
                st.session_state.uploaded_image_bytes = None
                st.session_state.uploaded_image_type = None
                st.rerun()

# Empty State Welcome Screen (when there are no chat messages)
if not st.session_state.messages:
    user_first_name = st.session_state.user_name.split()[0]
    st.markdown(f"""
        <div style="padding: 24px 0 10px 0;">
            <div style="font-size: 2.7rem; font-weight: 700; background: linear-gradient(90deg, #4285F4 0%, #A87FFB 50%, #FF8A80 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Hello, {user_first_name}
            </div>
            <div style="color: var(--text-primary); font-size: 1.45rem; font-weight: 500; margin-bottom: 24px;">
                How can I help you navigate Indian Standards today?
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.85rem; color:#8AB4F8; font-weight:600; margin-bottom:12px;'>SUGGESTED QUESTIONS</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍  Which standard applies to electric irons & appliances?", use_container_width=True):
            st.session_state.pending_query = "Which standard applies to electric appliances like electric irons?"
            st.rerun()

        if st.button("💎  How do I verify a 6-digit HUID hallmark on gold jewellery?", use_container_width=True):
            st.session_state.pending_query = "How do I verify a 6-digit HUID hallmark on gold jewellery?"
            st.rerun()

    with col2:
        if st.button("📋  What is the process to apply for a BIS licence?", use_container_width=True):
            st.session_state.pending_query = "What is the process to apply for a certification licence?"
            st.rerun()

        if st.button("🧸  What are the BIS certification rules for children's toys?", use_container_width=True):
            st.session_state.pending_query = "What are the BIS certification rules for toys?"
            st.rerun()

# Render Previous Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], caption="Attached Media", width=220)
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="✨"):
            intent = msg.get("intent", "other")
            intent_label = msg.get("intent_label", intent)
            model_used = msg.get("model_used")
            
            badge_class = f"intent-{intent}"
            model_html = f"<span class='top-bar-model-chip' style='margin-left:8px;'>{model_used}</span>" if model_used else ""

            st.markdown(
                f"<div style='display:flex; align-items:center; flex-wrap:wrap; margin-bottom:8px;'>"
                f"<span class='intent-badge {badge_class}'>● {intent_label}</span>"
                f"{model_html}"
                f"</div>",
                unsafe_allow_html=True
            )

            if msg.get("error"):
                st.warning(msg["error"])

            st.markdown(msg["content"])

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved BIS Sources ({len(sources)} documents)", expanded=False):
                    for idx, s in enumerate(sources):
                        st.markdown(f"""
                            <div class="source-box">
                                <div style="display:flex; justify-content:space-between; font-weight:600; color:#8AB4F8;">
                                    <span>📄 {s.get('source', 'Unknown')} (Page {s.get('page', 1)})</span>
                                    <span style="color:#A87FFB;">Match Score: {s.get('score', 0.0):.2f}</span>
                                </div>
                                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:2px;">Category: {s.get('category', 'general').capitalize()}</div>
                                <div style="font-size:0.83rem; margin-top:6px; line-height:1.45;">{s.get('text', '')}</div>
                            </div>
                        """, unsafe_allow_html=True)

# Query Input Execution
prompt_from_input = st.chat_input("Ask about Indian Standards (IS), ISI certification, or HUID hallmarking...")
query_to_process = None

if prompt_from_input:
    query_to_process = prompt_from_input
elif st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None

if query_to_process:
    active_img_bytes = st.session_state.uploaded_image_bytes
    active_img_type = st.session_state.uploaded_image_type

    # 1. Add User Message
    st.session_state.messages.append({
        "role": "user",
        "content": query_to_process,
        "image_bytes": active_img_bytes,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    with st.chat_message("user", avatar="👤"):
        if active_img_bytes:
            st.image(active_img_bytes, caption="Attached Media", width=220)
        st.write(query_to_process)

    # 2. Process via RAG & Gemini
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Analyzing Indian Standards repository & querying Gemini..."):
            history_context = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            res = answer(
                query=query_to_process,
                gemini_model=st.session_state.gemini_model,
                image_bytes=active_img_bytes,
                mime_type=active_img_type,
                chat_history=history_context,
                use_intent=st.session_state.use_intent_routing
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "intent": res["intent"],
                "intent_label": res["intent_label"],
                "sources": res["sources"],
                "model_used": res.get("model_used"),
                "error": res.get("error"),
                "timestamp": datetime.now().strftime("%H:%M")
            })

    # Clear active media
    st.session_state.uploaded_image_bytes = None
    st.session_state.uploaded_image_type = None

    st.rerun()
