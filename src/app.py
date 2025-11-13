# ----------------------------
# Project: Book Scraper - Scrape & Explore
# Author: Doufu (Github: DubuV2)
# Date: 2025-11-12
# Descrption: A script to scrape book data from a website to train myself, with options for detailed info, output formats
# ----------------------------

import os
import sqlite3
import pandas as pd 
import streamlit as st
import logging, queue
import time
import platform
from io import BytesIO
from logging.handlers import QueueHandler
from types import SimpleNamespace
from scraper import scrape_all_books

st.set_page_config(page_title="Book Explorer (Scrape + Explore)", layout="wide")

# --------- Utils ---------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

@st.cache_data
def load_df(csv_path: str | None, json_path: str | None, db_path: str | None) -> pd.DataFrame:
    """
    Loads data from CSV, JSON, or SQLite database.
    Performs data cleaning/type conversion (Price, Rating, Stock Extraction).
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif json_path and os.path.exists(json_path):
        df = pd.read_json(json_path)
    elif db_path and os.path.exists(db_path):
        con = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM books", con)
        con.close()
    else:
        raise FileNotFoundError("No valid data source found.")
    
    # Type conversion
    for col in ("Price", "Price (incl. tax)", "Price (excl. tax)"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if "Star Rating" in df.columns:
        df["Star Rating"] = pd.to_numeric(df["Star Rating"], errors='coerce')

    # Extract Stock from Availability
    if "Availability" in df.columns and "Stock" not in df.columns:
        df["Stock"] = (
            df["Availability"].astype(str)
            .str.extract(r"\((\d+)\s+available\)", expand=False)
            .astype("float")
            .fillna(0)
            .astype(int)
        )
    return df

def resolve_outputs(base_path: str, fmt: str):
    """
    Utility function to generate CSV and/or JSON paths based on the format.
    """
    base, ext = os.path.splitext(base_path)
    ext = ext.lower()
    if fmt == "csv":
        return (base + ".csv" if ext != ".csv" else base_path, None)
    if fmt == "json":
        return (None, base + ".json" if ext != ".json" else base_path)
    # both
    if ext in (".csv", ".json"):
        base = base
    return (base + ".csv", base + ".json")

def build_exports(df):
    """Build CSV and Excel exports with fallback engines"""
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Try both engines (XlsxWriter, then openpyxl)
    xlsx_buf = BytesIO()
    excel_ok = True
    try:
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="books", index=False)
    except ImportError:
        try:
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="books", index=False)
        except ImportError:
            excel_ok = False
    if excel_ok:
        xlsx_buf.seek(0)
        return csv_bytes, xlsx_buf
    else:
        return csv_bytes, None

# --- Constants ---
DEFAULT_DELAY = 0.3
DEFAULT_MAX_WORKERS = 10
DEFAULT_LIMIT = 0  
DEFAULT_DB_PATH = ""

# --------- UI ---------
st.title("Books Explorer")

# --- Developer mode toggle ---
SHOW_DB = st.sidebar.checkbox("Developer mode", value=False)
if SHOW_DB:
    # Developer mode
    log_level_name = "DEBUG"
    st.sidebar.caption("Developer mode: logging = DEBUG, DB viewer enabled.")
else:
    log_level_name = "INFO"

# Tab definitions (Dynamically include 'Database' in dev mode)
if SHOW_DB:
    tab_scrape, tab_explore, tab_db = st.tabs(["Scraper", "Explorer", "Database"])
else:
    tab_scrape, tab_explore = st.tabs(["Scraper", "Explorer"])

with tab_scrape:
    st.subheader("Start Scraping")
    
# --- Scraper Configuration ---
    col1, col2 = st.columns(2)
    with col1:
        base_url = st.text_input("Start URL", "http://books.toscrape.com/")
        if platform.system() == "Windows":
            default_path = os.path.expanduser("~/Desktop/books")
        else:
            default_path = os.path.join(os.path.dirname(__file__), "..", "data", "books")

        output_base = st.text_input(
            "Output Base Path",
            value=default_path,
            help="Where to save scraped files"
        )
        if not output_base.strip():
            st.warning("Please provide a valid output base path.")
            st.stop()
        
        output_base = os.path.abspath(output_base)
        ensure_dir(output_base)

    with col2:
        out_fmt = st.selectbox("Output Format", ["csv", "json", "both"], index=0)
        details = st.checkbox("Show Scrape Details", value=True)

    # Advanced Options (only in dev mode)
    if SHOW_DB:
        with st.expander("Advanced Options", expanded=False):
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                delay = st.slider("Delay (s)", 0.0, 5.0, DEFAULT_DELAY, 0.1)
            with a2:
                max_workers = st.number_input("Threads", 1, 64, DEFAULT_MAX_WORKERS, 1)
            with a3:
                limit = st.number_input("Max pages (0=all)", 0, 1000, DEFAULT_LIMIT, 1)
            with a4:
                db_path = st.text_input("SQLite DB Path", DEFAULT_DB_PATH)
    else:
        delay = DEFAULT_DELAY
        max_workers = DEFAULT_MAX_WORKERS
        limit = DEFAULT_LIMIT
        db_path = DEFAULT_DB_PATH

        st.caption(
            f"Defaults · Delay: {delay}s · Threads: {max_workers} · Pages: "f"{'all' if limit == 0 else limit} · SQLite DB: {'none' if not db_path else db_path}"
        )
    run = st.button("Start Scraping")

    if run:
        csv_out, json_out = resolve_outputs(output_base, out_fmt)
        for p in [csv_out, json_out, db_path]:
            if p:
                ensure_dir(p)
        
        args = SimpleNamespace(
            url=base_url,
            output=(csv_out or json_out or output_base),
            delay=delay,
            details=bool(details),
            max_workers=int(max_workers),
            format=out_fmt,
            limit=int(limit),
            db=(db_path if db_path.strip() else None),
        )

        log_placeholder = st.empty()

        # --- Logging Setup for Streamlit ---
        # 1. Create a Queue to capture logs from the scraper's threads
        log_q = queue.Queue()
        qhandler = QueueHandler(log_q)
        logging.getLogger().addHandler(qhandler)
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level_name))
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logs_buffer = []

        # Log clearing button setup
        clear_col, _ = st.columns([1,5])
        if clear_col.button("Clear Logs"):
            logs_buffer.clear()
            log_placeholder.code("", language="text")

        def drain_log_queue():
            """
            Reads messages from th equeue and updates the log display.
            """
            drained = False
            while True:
                try:
                    record = log_q.get_nowait()
                except queue.Empty:
                    break
                drained = True
                msg = log_formatter.format(record)
                logs_buffer.append(msg)
            if drained:
                # Memory efficient: only keep last 200 logs
                if len(logs_buffer) > 200:
                    del logs_buffer[:-200]
                log_placeholder.code("\n".join(logs_buffer[-5:]), language="text")

        with st.status("Scraping...", expanded=True) as status:
            progress_bar = st.progress(0, text="Progress: 0%")
            st.write(f"URL: {args.url}")
            st.write(f"Format: {args.format} -> output: {args.output}")
            if args.db:
                st.write(f"DB: {args.db}")
            if SHOW_DB:
                st.write(f"details={args.details}, delay={args.delay}, max_workers={args.max_workers}, limit={args.limit}")
            else:
                st.write(f"details={args.details}")
            try:
                # Define progress bar ranges
                if args.details:
                    PAGES_END, LISTING_END, DETAILS_END = 30, 50, 98
                else:
                    PAGES_END, LISTING_END, DETAILS_END = 30, 98, 98

                # Call the generator and update UI on each yield
                for tick in scrape_all_books(args):
                    drain_log_queue()

                    if isinstance(tick, tuple):
                        progress_percent, meta = tick
                    else:
                        progress_percent, meta = tick, None

                    # Determine phase based on progress
                    if progress_percent < PAGES_END:
                        if meta and meta.get("phase") == "pages":
                            phase = f"Collecting pages... ({meta.get('i', 0)}/{meta.get('total', '?')})"
                        else:
                            phase = "Collecting pages..."
                    elif progress_percent < LISTING_END:
                        if meta and meta.get("phase") == "listing":
                            phase = f"Parsing collected pages... ({meta.get('i', 0)}/{meta.get('total', '?')})"
                        else:
                            phase = "Parsing collected pages..."
                    elif args.details and progress_percent < DETAILS_END:
                        if meta and isinstance(meta, dict):
                            phase = f"Fetching detailed book info... ({meta.get('i', 0)}/{meta.get('total', '?')})"
                        else:
                            phase = "Fetching detailed book info..."
                    else:
                        phase = "Saving results..."

                    progress_bar.progress(progress_percent, text=f"{phase} ({progress_percent}%)") 
                    time.sleep(0.05)

                drain_log_queue()
                status.update(label="Scraping completed successfully!", state="complete")
                # Final output summary
                paths = []
                if csv_out and os.path.exists(csv_out):
                    paths.append(f"CSV: {os.path.abspath(csv_out)}")
                if json_out and os.path.exists(json_out):
                    paths.append(f"JSON: {os.path.abspath(json_out)}")
                if args.db and os.path.exists(args.db):
                    paths.append(f"SQLite DB: {os.path.abspath(args.db)}")
                
                if paths:
                    st.success("Scraping finished. Files generated:")
                    st.caption("\n".join(paths))
                else:
                    st.warning("Scraping finished but no output files were found.")
                    
            except Exception as e:
                drain_log_queue()
                status.update(label=f"Error during scraping: {e}", state="error")
                st.exception(e)
            finally:
                # Clean up logging handler
                root_logger.removeHandler(qhandler)


with tab_explore:
    st.subheader("Explore Scraped Data")

    # Input fields for data file paths 
    default_base = "data/books"
    auto_csv, auto_json = resolve_outputs(default_base, "both")
    csv_guess = st.text_input("CSV Path", auto_csv)
    json_guess = st.text_input("JSON Path", auto_json)
    db_guess = st.text_input("SQLite DB Path", "data/books.db")

# --- Controls row ---
    col_load, col_export = st.columns([1, 1])

    with col_load:
        load_clicked = st.button("Load Data", use_container_width=True)
    
    if load_clicked:
        try:
            df = load_df(
                csv_guess if os.path.exists(csv_guess) else None,
                json_guess if os.path.exists(json_guess) else None,
                db_guess if os.path.exists(db_guess) else None,
            )
            st.success(f"Data loaded successfully! {len(df)} records found.")
            st.session_state["books_df"] = df
        except Exception as e:
            st.error(str(e))
    
    df = st.session_state.get("books_df")
    
    with col_export:
        if isinstance(df, pd.DataFrame) and not df.empty:
            csv_bytes, xlsx_buf = build_exports(df)

            try:
                with st.popover("Export Data", use_container_width=True):
                    st.caption("Choose format:")
                    st.download_button("CSV (.csv)", data=csv_bytes, file_name="filtered_books.csv", mime="text/csv")
                    if xlsx_buf is not None:
                        st.download_button(
                            "Excel (.xlsx)",
                            data=xlsx_buf,
                            file_name="filtered_books.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        st.info("Excel export not available (install XlsxWriter or openpyxl).")
            except AttributeError:
                # Fallback for older Streamlit versions
                with st.expander("Export Data"):
                    st.caption("Choose format:")
                    st.download_button("CSV (.csv)", data=csv_bytes, file_name="filtered_books.csv", mime="text/csv")
                    if xlsx_buf is not None:
                        st.download_button(
                            "Excel (.xlsx)",
                            data=xlsx_buf,
                            file_name="filtered_books.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        st.info("Excel export not available (install XlsxWriter or openpyxl).")
        else:
            st.button("Export Data", disabled=True, use_container_width=True)


    if isinstance(df, pd.DataFrame):
        st.caption(f"Columns: {list(df.columns)}")

        # --- Filters in Sidebar ---
        st.sidebar.header("Filters")
        cats = sorted(df["Category"].dropna().unique()) if "Category" in df else []
        sel_cats = st.sidebar.multiselect("Category", cats)
        ratings = sorted(df["Star Rating"].dropna().unique()) if "Star Rating" in df else []
        sel_ratings = st.sidebar.multiselect("Notes", ratings)
        if "Price" in df:
            pmin, pmax = float(df["Price"].min()), float(df["Price"].max())
            sel_price = st.sidebar.slider("Price Range", pmin, pmax, (pmin, pmax))
        else:
            sel_price = (None, None)
        only_in_stock = st.sidebar.checkbox("Stock > 0", value=False)
        q = st.sidebar.text_input("Search in Title/Description")

        # --- Filtering Logic ---
        view = df.copy()
        if sel_cats and "Category" in view:
            view = view[view["Category"].isin(sel_cats)]
        if sel_ratings and "Star Rating" in view:
            view = view[view["Star Rating"].isin(sel_ratings)]
        if sel_price[0] is not None and "Price" in view:
            view = view[(view["Price"] >= sel_price[0]) & (view["Price"] <= sel_price[1])]
        if only_in_stock and "Stock" in view:
            view = view[view["Stock"] > 0]
        if q:
            view = view[view["Title"].astype(str).str.contains(q, case=False, na=False)]
        
        # --- KPIs ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Books", f"{len(view)} / {len(df)}")
        if "Price" in view: c2.metric("Median Price", f"{view['Price'].median():.2f}")
        if "Star Rating" in view: c3.metric("Average Rating", f"{view['Star Rating'].mean():.2f}")

        # --- Data Display ---
        cols = [c for c in ["Title", "Category", "Price", "Star Rating", "Stock", "Availability", "Link", "Image"] if c in view.columns]
        st.dataframe(view[cols] if cols else view, use_container_width=True, hide_index=True)

        # --- Graphs ---
        colA, colB = st.columns(2)
        with colA:
            if "Category" in view and "Price" in view and not view.empty:
                st.subheader("Price by Category (Top 10)")
                by_cat = view.groupby("Category")["Price"].median().sort_values(ascending=False).head(10)
                st.bar_chart(by_cat)
        with colB:
            if "Star Rating" in view and not view.empty:
                st.subheader("Books by Star Rating")
                st.bar_chart(view["Star Rating"].value_counts().sort_index())

# --- Database Viewer (Developer Mode) ---
if SHOW_DB:
    with tab_db:
        st.subheader("Database Viewer")
        db_path_view = st.text_input("SQLite DB Path", "data/books.db", key="db_path_view")
        if not os.path.exists(db_path_view):
            st.warning(f"Database file '{db_path_view}' does not exist.")
        else:
            con = sqlite3.connect(db_path_view)
            try:
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", con)
                table_names = tables["name"].tolist()
                if not table_names:
                    st.info("No tables found in the database.")
                else:
                    sel_table = st.selectbox("Select table", table_names)
                    limit = st.slider("Preview rows", 5, 1000, 50, 5)
                    df_preview = pd.read_sql(f"SELECT * FROM {sel_table} LIMIT {limit};", con)
                    st.caption(f"Rows in preview: {len(df_preview)}")
                    st.dataframe(df_preview, use_container_width=True, hide_index=True)

                    st.divider()
                    st.write("Run a **read-only** SQL query (SELECT only):")
                    q = st.text_area("Query", f"SELECT  COUNT(*) AS total FROM {sel_table};")
                    if st.button("Execute", type="primary"):
                        q_stripped = q.strip().lower()
                        if not q_stripped.startswith("select"):
                            st.error("Only SELECT queries are allowed.")
                        else:
                            try:
                                df_query = pd.read_sql(q, con)
                                st.success(f"Query executed successfully! {len(df_query)} records found.")
                                st.dataframe(df_query, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Error executing query: {e}")
            finally:
                con.close()