"""
BudgetWise AI — Single-file Streamlit application
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import joblib
import plotly.express as px
import plotly.graph_objects as go  # Added for better custom charts
import hashlib
from typing import Optional, Tuple
import google.generativeai as genai

# ------------------------
# CONFIG
# ------------------------
DB_PATH = Path("budget_app.db")
IMAGE_DIR = Path("expense_images")
MODEL_PATH = Path("models/best_finance_model.pkl")
# Ensure directories exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

# ------------------------
# STYLES (Stable & Professional Dark Theme)
# ------------------------
st.set_page_config(page_title="BudgetWise AI", page_icon="💰", layout="wide")

CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

<style>

/* --- GLOBAL RESET / BASE STYLING --- */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif !important;
}

body {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

.stApp {
    background-color: #0d1117 !important;
}

/* Prevent card overlapping */
.block-container {
    padding-top: 2rem !important;
}

/* --- HEADER --- */
.header {
    padding: 1.5rem 2rem;
    border-radius: 0.75rem;
    background-color: #161b22;
    border: 1px solid #30363d;
    margin-bottom: 2rem;
}
.title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.subtitle {
    font-size: 1rem;
    color: #8b949e;
}

/* --- GENERAL CONTENT CARDS --- */
.card {
    border-radius: 0.75rem;
    padding: 1.5rem;
    background-color: #161b22;
    border: 1px solid #30363d;
    margin-bottom: 1.5rem;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.25);
}

/* --- METRIC CARDS (Dashboard) --- */
.metric-card {
    border-radius: 0.75rem;
    padding: 1rem;
    background-color: #161b22;
    border: 1px solid #30363d;
    color: #c9d1d9;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.25);
}
.metric-card h3 {
    font-size: 1.45rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0 0 0.5rem 0;
}
.metric-card .small-muted {
    color: #8b949e;
    font-size: 0.875rem;
}

/* --- BUTTONS --- */
.stButton>button {
    border-radius: 0.5rem !important;
    padding: 0.55rem 1rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    background-color: #238636 !important;
    border: 1px solid #2ea043 !important;
    transition: 0.25s ease;
}
.stButton>button:hover {
    background-color: #2ea043 !important;
    border-color: #3fb950 !important;
    transform: translateY(-2px);
}

/* --- SIDEBAR --- */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .css-1d391kg {
    padding-top: 2rem !important;
}

/* --- DATAFRAMES --- */
.stDataFrame, .dataframe {
    border: 1px solid #30363d !important;
    border-radius: 0.5rem !important;
    background-color: #0d1117 !important;
}

/* --- INPUT FIELDS (modern style) --- */
input, textarea, select {
    border-radius: 0.5rem !important;
    background-color: #0f1620 !important;
    color: #ffffff !important;
    border: 1px solid #30363d !important;
}

/* fix overlapping select and date inputs */
.css-1uixxvy, .css-qrbaxs {
    margin-bottom: 1rem !important;
}

/* --- PLOTLY CHARTS --- */
.js-plotly-plot {
    background-color: transparent !important;
}

/* --- RADIO BUTTON FIX (login card) --- */
.stRadio > label {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------
# HELPER: GRAPH STYLING
# ------------------------
def style_plotly_chart(fig):
    """Applies a consistent dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9', family="Inter, sans-serif"),
        margin=dict(t=30, l=10, r=10, b=10),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#30363d', 
            zeroline=False,
            showline=True,
            linecolor='#30363d'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#30363d', 
            zeroline=False,
            showline=True,
            linecolor='#30363d'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
    )
    return fig

# ------------------------
# DATABASE & CORE LOGIC
# ------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def ensure_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with ensure_db_connection() as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS expenses (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, date TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL, notes TEXT, created_at TEXT NOT NULL, image_path TEXT, FOREIGN KEY(user_id) REFERENCES users(id))")
        c.execute("CREATE TABLE IF NOT EXISTS budgets (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, month TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL, UNIQUE(user_id, month, category), FOREIGN KEY(user_id) REFERENCES users(id))")
        c.execute("CREATE TABLE IF NOT EXISTS recurring_expenses (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, description TEXT, amount REAL NOT NULL, category TEXT NOT NULL, frequency TEXT NOT NULL, start_date TEXT NOT NULL, next_due_date TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))")
        conn.commit()

init_db()

def add_user(username, password):
    with ensure_db_connection() as conn:
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
            conn.commit()
            return True, "User created successfully."
        except sqlite3.IntegrityError:
            return False, "Username already exists."

def login_user(username, password):
    with ensure_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
        user = cursor.fetchone()
        return (True, user['id']) if user else (False, None)

def add_expense(user_id, dt, category, amount, notes, image_path):
    with ensure_db_connection() as conn:
        conn.execute("INSERT INTO expenses (user_id, date, category, amount, notes, created_at, image_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (user_id, dt.isoformat(), category, amount, notes, datetime.utcnow().isoformat(), image_path))
        conn.commit()

def get_expenses(user_id):
    with ensure_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM expenses WHERE user_id = ? ORDER BY date DESC, id DESC", conn, params=(user_id,))
    if not df.empty:
        df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def delete_expense(expense_id, user_id):
    with ensure_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM expenses WHERE id = ? AND user_id = ?", (expense_id, user_id))
        result = cursor.fetchone()
        if result and result['image_path']:
            try:
                Path(result['image_path']).unlink(missing_ok=True)
            except Exception as e:
                st.toast(f"Error deleting image file: {e}")
        cursor.execute("DELETE FROM expenses WHERE id = ? AND user_id = ?", (expense_id, user_id))
        conn.commit()

def upsert_budget(user_id, month, category, amount):
    with ensure_db_connection() as conn:
        conn.execute("INSERT INTO budgets (user_id, month, category, amount) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, month, category) DO UPDATE SET amount=excluded.amount",
                     (user_id, month, category, amount))
        conn.commit()

def get_budgets(user_id, month=None):
    with ensure_db_connection() as conn:
        if month:
            df = pd.read_sql_query("SELECT * FROM budgets WHERE user_id = ? AND month = ?", conn, params=(user_id, month))
        else:
            df = pd.read_sql_query("SELECT * FROM budgets WHERE user_id = ?", conn, params=(user_id,))
    return df

def add_recurring_expense(user_id, description, amount, category, frequency, start_date):
    with ensure_db_connection() as conn:
        conn.execute("INSERT INTO recurring_expenses (user_id, description, amount, category, frequency, start_date, next_due_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (user_id, description, amount, category, frequency, start_date.isoformat(), start_date.isoformat()))
        conn.commit()

def get_recurring_expenses(user_id):
    with ensure_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM recurring_expenses WHERE user_id = ?", conn, params=(user_id,))
    if not df.empty:
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date
        df['next_due_date'] = pd.to_datetime(df['next_due_date']).dt.date
    return df

def delete_recurring_expense(rec_id, user_id):
    with ensure_db_connection() as conn:
        conn.execute("DELETE FROM recurring_expenses WHERE id = ? AND user_id = ?", (rec_id, user_id))
        conn.commit()

def process_recurring_expenses(user_id):
    today = date.today()
    recs = get_recurring_expenses(user_id)
    if recs.empty: return 0
    
    conn = ensure_db_connection()
    inserted_count = 0
    for _, row in recs.iterrows():
        next_due = row['next_due_date']
        while next_due <= today:
            add_expense(user_id, next_due, row['category'], row['amount'], f"Recurring: {row['description']}", None)
            inserted_count += 1
            if row['frequency'] == 'daily': next_due += timedelta(days=1)
            elif row['frequency'] == 'weekly': next_due += timedelta(weeks=1)
            elif row['frequency'] == 'monthly':
                year, month = (next_due.year, next_due.month + 1) if next_due.month < 12 else (next_due.year + 1, 1)
                day = min(next_due.day, 28)
                next_due = date(year, month, day)
            elif row['frequency'] == 'yearly': next_due = date(next_due.year + 1, next_due.month, next_due.day)
            else: break
        
        conn.execute("UPDATE recurring_expenses SET next_due_date = ? WHERE id = ?", (next_due.isoformat(), row['id']))
    conn.commit()
    conn.close()
    return inserted_count

@st.cache_resource(show_spinner=False)
def load_prediction_model():
    if not MODEL_PATH.exists(): return None
    try: return joblib.load(MODEL_PATH)
    except Exception as e: st.error(f"Error loading model: {e}"); return None

prediction_model = load_prediction_model()

def predict_amount(dt: date, category: str) -> float:
    if prediction_model is None: return 50.0
    try:
        df = pd.DataFrame([{"category": category, "Year": dt.year, "Month": dt.month, "Day": dt.day}])
        return abs(float(prediction_model.predict(df)[0]))
    except Exception: return 50.0

if 'logged_in' not in st.session_state: st.session_state.logged_in = False

def do_logout():
    st.session_state.clear()
    st.session_state.logged_in = False
    st.rerun()

# ------------------------
# UI PAGES
# ------------------------
def auth_page():
    st.markdown('<div class="header"><p class="title">💰 BudgetWise AI</p><p class="subtitle">Your intelligent financial companion</p></div>', unsafe_allow_html=True)
    mode = st.radio("Access Your Account", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
    
    if mode == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                ok, uid = login_user(username.strip(), password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user_id = uid
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    else:
        with st.form("register_form"):
            username = st.text_input("Choose Username")
            password = st.text_input("Choose Password", type="password")
            password2 = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register", use_container_width=True):
                if password != password2: st.error("Passwords do not match.")
                elif not all([username, password]): st.error("All fields are required.")
                else:
                    ok, msg = add_user(username.strip(), password)
                    if ok: st.success(msg + " Please login.")
                    else: st.error(msg)

def sidebar_after_login():
    st.sidebar.markdown(f"### Welcome, {st.session_state.username}!")
    page = st.sidebar.radio("Navigate", ["📊 Dashboard", "➕ Add Expense", "📜 History", "💰 Budgets", "🔁 Recurring", "🤖 AI Predictions", "⭐ AI Financial Advisor", "⬆️ Import / Export", "⚙️ Settings"])
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"): do_logout()
    return page

def dashboard_page():
    st.markdown('<div class="card"><h2>📊 Dashboard</h2></div>', unsafe_allow_html=True)
    df = get_expenses(st.session_state.user_id)
    if df.empty:
        st.info("Welcome! Add your first expense to see your dashboard.")
        return

    # Basic stats
    this_month_start = date.today().replace(day=1)
    df['date_obj'] = pd.to_datetime(df['date']) # Ensure datetime type
    df_this_month = df[df['date_obj'].dt.date >= this_month_start]
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h3>₹{df['amount'].sum():,.2f}</h3><p class='small-muted'>Total Spent</p></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>{len(df)}</h3><p class='small-muted'>Transactions</p></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>₹{df_this_month['amount'].sum():,.2f}</h3><p class='small-muted'>This Month</p></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h3>₹{df.groupby('date')['amount'].sum().mean():,.2f}</h3><p class='small-muted'>Avg. Daily Spend</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    # --- CHART 1: DONUT CHART (Spending by Category) ---
    with col1:
        st.markdown("#### Spending by Category")
        if df_this_month.empty:
            st.caption("No data for this month to display chart.")
        else:
            by_cat = df_this_month.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=False)
            
            # Create interactive Donut Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=by_cat['category'],
                values=by_cat['amount'],
                hole=.6,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(colors=px.colors.sequential.Blues_r),
                hovertemplate="<b>%{label}</b><br>₹%{value:,.2f}<extra></extra>"
            )])
            
            fig_pie = style_plotly_chart(fig_pie)
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- CHART 2: AREA CHART (Spending Trend) ---
    with col2:
        st.markdown("#### Spending Trend (Last 6 Months)")
        
        # Get data for the last 6 months
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        df_trend = df[df['date_obj'] >= six_months_ago].copy()
        
        if df_trend.empty:
            st.caption("Not enough data for trend analysis.")
        else:
            # Group by Month and sort chronologically
            df_trend['month_year'] = df_trend['date_obj'].dt.to_period('M')
            monthly_trend = df_trend.groupby('month_year')['amount'].sum().reset_index()
            monthly_trend['month_year'] = monthly_trend['month_year'].dt.to_timestamp()
            monthly_trend = monthly_trend.sort_values('month_year')

            # Create Smooth Area Chart
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=monthly_trend['month_year'],
                y=monthly_trend['amount'],
                mode='lines+markers',
                fill='tozeroy',  # Fill area below line
                name='Spent',
                line=dict(color='#58a6ff', width=3, shape='spline'), # Smooth spline
                marker=dict(size=8, color='#1f6feb', line=dict(width=2, color='white')),
                hovertemplate="<b>%{x|%B %Y}</b><br>₹%{y:,.2f}<extra></extra>"
            ))
            
            fig_trend = style_plotly_chart(fig_trend)
            fig_trend.update_layout(xaxis_title=None, yaxis_title="Amount (₹)")
            st.plotly_chart(fig_trend, use_container_width=True)

def add_expense_page():
    st.markdown('<div class="card"><h2>➕ Add New Expense</h2></div>', unsafe_allow_html=True)
    df = get_expenses(st.session_state.user_id)
    default_cats = ["Food", "Transport", "Shopping", "Rent", "Utilities", "Entertainment", "Health", "Other"]
    user_cats = sorted(df['category'].unique()) if not df.empty else []
    all_cats = sorted(set(default_cats + user_cats))

    with st.form("add_expense_form"):
        col1, col2 = st.columns(2)
        with col1:
            dt = st.date_input("Date", value=date.today())
            category = st.selectbox("Category", options=all_cats)
            amount = st.number_input("Amount (₹)", min_value=0.01, format="%.2f")
        with col2:
            notes = st.text_area("Notes (Optional)")
            uploaded_file = st.file_uploader("Upload Receipt (Optional)", type=["png", "jpg", "jpeg"])
        
        if st.form_submit_button("Add Expense", use_container_width=True):
            image_path = None
            if uploaded_file:
                fname = f"receipt_{st.session_state.user_id}_{int(datetime.now().timestamp())}{Path(uploaded_file.name).suffix}"
                image_path = str(IMAGE_DIR / fname)
                with open(image_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            add_expense(st.session_state.user_id, dt, category, amount, notes, image_path)
            st.success("Expense added successfully!")

def history_page():
    st.markdown('<div class="card"><h2>📜 Transaction History</h2></div>', unsafe_allow_html=True)
    df = get_expenses(st.session_state.user_id)
    if df.empty:
        st.info("No expenses found.")
        return

    fdf = df.copy()
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        cats = ["All"] + sorted(fdf['category'].unique().tolist())
        cat_sel = st.selectbox("Filter by Category", options=cats)
        if cat_sel != "All": fdf = fdf[fdf['category'] == cat_sel]
    with col2:
        date_range = st.date_input("Filter by Date Range", (fdf['date'].min(), fdf['date'].max()))
        if len(date_range) == 2: fdf = fdf[(fdf['date'] >= date_range[0]) & (fdf['date'] <= date_range[1])]
    with col3:
        keyword = st.text_input("Search in Notes")
        if keyword: fdf = fdf[fdf['notes'].str.contains(keyword, case=False, na=False)]

    st.dataframe(fdf[['id', 'date', 'category', 'amount', 'notes']], use_container_width=True)
    st.markdown("#### Delete Expenses")
    ids_to_delete = st.multiselect("Select expense IDs to delete", options=fdf['id'].tolist())
    if st.button("Delete Selected") and ids_to_delete:
        for eid in ids_to_delete: delete_expense(eid, st.session_state.user_id)
        st.success(f"Deleted {len(ids_to_delete)} expenses."); st.rerun()

def budgets_page():
    st.markdown('<div class="card"><h2>💰 Budgets</h2></div>', unsafe_allow_html=True)
    month_str = st.date_input("Select Month for Budget", date.today()).strftime("%Y-%m")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Set Monthly Budget")
        df = get_expenses(st.session_state.user_id)
        cats = sorted(df['category'].unique().tolist()) if not df.empty else ["General"]
        with st.form("budget_form"):
            category = st.selectbox("Category", cats)
            amount = st.number_input("Budget Amount (₹)", min_value=0.0)
            if st.form_submit_button("Save Budget"):
                upsert_budget(st.session_state.user_id, month_str, category, amount)
                st.success("Budget saved!"); st.rerun()

    with col2:
        st.markdown(f"#### Budget vs Actual for {month_str}")
        budgets_df = get_budgets(st.session_state.user_id, month=month_str)
        if budgets_df.empty:
            st.info("No budgets set for this month.")
        else:
            month_expenses = df[pd.to_datetime(df['date']).dt.strftime('%Y-%m') == month_str]
            actuals = month_expenses.groupby('category')['amount'].sum().reset_index().rename(columns={'amount': 'actual'})
            merged = pd.merge(budgets_df, actuals, on='category', how='left').fillna(0)
            merged['remaining'] = merged['amount'] - merged['actual']
            st.dataframe(merged[['category', 'amount', 'actual', 'remaining']], use_container_width=True)

def recurring_page():
    st.markdown('<div class="card"><h2>🔁 Recurring Expenses</h2></div>', unsafe_allow_html=True)
    with st.form("add_recurring"):
        st.markdown("#### Add New Recurring Expense")
        desc = st.text_input("Description")
        amount = st.number_input("Amount (₹)", min_value=0.01)
        category = st.selectbox("Category", ["Rent", "Subscription", "Bills", "Salary", "Other"])
        frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly", "yearly"])
        start_date = st.date_input("Start Date", date.today())
        if st.form_submit_button("Add Recurring Expense"):
            add_recurring_expense(st.session_state.user_id, desc, amount, category, frequency, start_date)
            st.success("Recurring expense added."); st.rerun()

    recs = get_recurring_expenses(st.session_state.user_id)
    if not recs.empty:
        st.markdown("#### Active Recurring Expenses")
        st.dataframe(recs[['id', 'description', 'amount', 'category', 'frequency', 'next_due_date']], use_container_width=True)
        del_id = st.number_input("Enter ID to delete", min_value=0, step=1, value=None)
        if st.button("Delete") and del_id:
            delete_recurring_expense(del_id, st.session_state.user_id)
            st.success(f"Deleted ID {del_id}"); st.rerun()

def ai_predictions_page():
    st.markdown('<div class="card"><h2>🤖 AI Expense Forecast</h2></div>', unsafe_allow_html=True)
    df = get_expenses(st.session_state.user_id)
    if df.empty or prediction_model is None:
        st.info("Add more expenses or check model status to enable predictions.")
        return

    cats = sorted(df['category'].unique().tolist())
    
    col_input, col_graph = st.columns([1, 3])
    
    with col_input:
        st.markdown("#### Configuration")
        category = st.selectbox("Select Category", cats)
        days = st.slider("Forecast Days", 7, 90, 30)
        generate_btn = st.button("Generate Forecast", use_container_width=True)

    if generate_btn:
        with col_graph:
            # 1. Get Historical Data (Last 30 entries for context)
            df['date_obj'] = pd.to_datetime(df['date'])
            cat_history = df[df['category'] == category].sort_values('date_obj')
            
            # 2. Generate Future Dates & Predictions
            future_dates = [date.today() + timedelta(days=i) for i in range(1, days + 1)]
            preds = [predict_amount(d, category) for d in future_dates]
            
            # 3. Build the Visualization
            fig = go.Figure()

            # Trace 1: Historical Data (Solid Line)
            if not cat_history.empty:
                # Group by date to handle multiple transactions per day
                hist_grouped = cat_history.groupby('date_obj')['amount'].sum().reset_index()
                # Limit to recent history for clarity (last 30 entries)
                hist_grouped = hist_grouped.tail(30)
                
                fig.add_trace(go.Scatter(
                    x=hist_grouped['date_obj'], 
                    y=hist_grouped['amount'],
                    mode='lines+markers',
                    name='Actual History',
                    line=dict(color='#238636', width=3),
                    marker=dict(size=6)
                ))
                
                # Connect the last history point to the first prediction point visually
                last_hist_date = hist_grouped['date_obj'].iloc[-1]
                last_hist_val = hist_grouped['amount'].iloc[-1]
                
                # Prepend the last actual data point to predictions to close the gap
                future_dates_plot = [last_hist_date] + future_dates
                preds_plot = [last_hist_val] + preds
            else:
                future_dates_plot = future_dates
                preds_plot = preds

            # Trace 2: Prediction (Dashed Line)
            fig.add_trace(go.Scatter(
                x=future_dates_plot, 
                y=preds_plot,
                mode='lines+markers',
                name='AI Forecast',
                line=dict(color='#a371f7', width=3, dash='dash'), # Purple dashed line
                marker=dict(size=6, symbol='diamond')
            ))

            fig.update_layout(
                title=f"Spending Forecast: {category}",
                xaxis_title="Date",
                yaxis_title="Amount (₹)",
                hovermode="x unified"
            )
            
            fig = style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

def ai_financial_advisor_page():
    st.markdown('<div class="card"><h2>⭐ AI Financial Advisor</h2><p class="subtitle">Get personalized financial advice from Gemini Pro</p></div>', unsafe_allow_html=True)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # Using a standard model name
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception:
        st.error("Could not configure Gemini API. Ensure `secrets.toml` has a valid GEMINI_API_KEY.")
        return

    df = get_expenses(st.session_state.user_id)
    if df.empty:
        st.info("Add some expenses to get financial advice."); return

    if st.button("✨ Get AI Financial Advice", use_container_width=True):
        with st.spinner("Your AI advisor is analyzing your finances..."):
            df_recent = df[df['date'] >= date.today() - timedelta(days=30)]
            if df_recent.empty:
                st.warning("No expenses in the last 30 days to analyze."); return

            summary = f"Total Spent (Last 30 Days): ₹{df_recent['amount'].sum():,.2f}\nTop Categories:\n"
            summary += "\n".join([f"- {cat}: ₹{amt:,.2f}" for cat, amt in df_recent.groupby('category')['amount'].sum().nlargest(5).items()])
            prompt = f"You are 'BudgetWise AI', a financial advisor for a user in India. Analyze the spending summary below and provide actionable advice in Markdown. Use ₹. Be encouraging.\n\nData:\n{summary}\n\nProvide tips on savings and budgeting."
            try:
                response = model.generate_content(prompt)
                st.markdown("---"); st.markdown(response.text)
            except Exception as e:
                st.error(f"Could not generate advice: {e}")

def import_export_page():
    st.markdown('<div class="card"><h2>⬆️ Import / Export Data</h2></div>', unsafe_allow_html=True)
    df = get_expenses(st.session_state.user_id)
    if not df.empty:
        st.download_button("Download All Expenses (CSV)", df.to_csv(index=False).encode('utf-8'), "expenses.csv", "text/csv")
    
    st.markdown("#### Import from CSV")
    uploaded_file = st.file_uploader("CSV must have 'date', 'category', 'amount' columns.", type="csv")
    if uploaded_file:
        try:
            import_df = pd.read_csv(uploaded_file)
            if not {'date', 'category', 'amount'}.issubset(import_df.columns):
                st.error("CSV is missing required columns."); return
            for _, row in import_df.iterrows():
                add_expense(st.session_state.user_id, pd.to_datetime(row['date']).date(), row['category'], row['amount'], row.get('notes', ''))
            st.success(f"Imported {len(import_df)} expenses."); st.rerun()
        except Exception as e:
            st.error(f"Failed to import: {e}")

def settings_page():
    st.markdown('<div class="card"><h2>⚙️ Settings</h2></div>', unsafe_allow_html=True)
    st.write(f"**Username:** {st.session_state['username']}")
    st.markdown("#### Change Password")
    with st.form("change_password"):
        current_pw = st.text_input("Current Password", type="password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        if st.form_submit_button("Change Password"):
            if new_pw != confirm_pw: st.error("New passwords do not match.")
            else:
                ok, _ = login_user(st.session_state['username'], current_pw)
                if ok:
                    with ensure_db_connection() as conn:
                        conn.execute("UPDATE users SET password = ? WHERE id = ?", (hash_password(new_pw), st.session_state.user_id))
                        conn.commit()
                    st.success("Password updated.")
                else:
                    st.error("Incorrect current password.")

def main():
    if not st.session_state.logged_in:
        auth_page(); return

    if st.session_state.user_id:
        try: process_recurring_expenses(st.session_state.user_id)
        except Exception: pass
    
    page = sidebar_after_login()
    
    pages = {
        "📊 Dashboard": dashboard_page, "➕ Add Expense": add_expense_page, "📜 History": history_page,
        "💰 Budgets": budgets_page, "🔁 Recurring": recurring_page, "🤖 AI Predictions": ai_predictions_page,
        "⭐ AI Financial Advisor": ai_financial_advisor_page, "⬆️ Import / Export": import_export_page,
        "⚙️ Settings": settings_page,
    }
    pages.get(page, dashboard_page)()

if __name__ == "__main__":
    main()
