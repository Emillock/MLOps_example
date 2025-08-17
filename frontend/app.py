import io
import json
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="DataMinds'25 - ML Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖",
)


def detect_mime(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return "text/csv"
    if name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if name.endswith(".xls"):
        return "application/vnd.ms-excel"
    if name.endswith(".parquet"):
        return "application/vnd.apache.parquet"
    return "application/octet-stream"


def load_df_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame | None:
    """Load CSV or Excel bytes into DataFrame (for preview)."""
    try:
        bio = io.BytesIO(file_bytes)
        if filename.lower().endswith(".csv"):
            return pd.read_csv(bio)
        elif filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(bio)
        elif filename.lower().endswith(".parquet"):
            return pd.read_parquet(bio)
        else:
            st.error(
                "Unsupported file format. Please upload a CSV, Excel, or Parquet file."
            )
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def send_to_api(file_bytes: bytes, filename: str, api_url: str) -> dict | None:
    """Send the raw uploaded file to FastAPI /predict."""
    try:
        files = {"file": (filename, file_bytes, detect_mime(filename))}
        resp = requests.post(api_url, files=files, timeout=60)
        # FastAPI: 200 OK on success; 4xx/5xx otherwise with JSON detail
        if resp.headers.get("content-type", "").startswith("application/json"):
            data = resp.json()
        else:
            st.error(
                f"Unexpected response from API (status {resp.status_code}): {resp.text[:400]}"
            )
            return None

        if resp.status_code == 200:
            return data
        # Error path from FastAPI (HTTPException)
        detail = data.get("detail") if isinstance(data, dict) else None
        st.error(f"API Error {resp.status_code}: {detail or data}")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while calling API: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def main():
    # Session state
    if "uploaded_name" not in st.session_state:
        st.session_state.uploaded_name = None  # store filename here
    if "file_bytes" not in st.session_state:
        st.session_state.file_bytes = None  # raw bytes (to send to API)
    if "df" not in st.session_state:
        st.session_state.df = None  # preview dataframe
    if "results" not in st.session_state:
        st.session_state.results = None  # API response (parsed)
    if "pred_ts" not in st.session_state:
        st.session_state.pred_ts = None

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
        <h2>ML Predictor</h2>
        <p>Upload your data and get predictions from the backend model.</p>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        api_url = "http://backend:8000/predict"

        st.markdown("### 📁 Upload Your Data")
        uploaded = st.file_uploader(
            "Choose a CSV, Excel, or Parquet file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Supported: .csv, .xlsx, .xls, .parquet",
        )

        # --- Robust upload handling: compare bytes rather than object identity ---
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            # if there are no bytes stored yet or the bytes differ, treat as a new file
            if (
                st.session_state.file_bytes is None
                or st.session_state.file_bytes != file_bytes
            ):
                st.session_state.uploaded_name = uploaded.name
                st.session_state.file_bytes = file_bytes
                st.session_state.df = load_df_from_bytes(file_bytes, uploaded.name)
                # new upload -> clear previous results (user changed input)
                st.session_state.results = None
                # reset pred timestamp so filename will be fresh when new predictions are made
                st.session_state.pred_ts = None
        else:
            # user removed file; clear stored state
            if (
                st.session_state.file_bytes is not None
                or st.session_state.uploaded_name is not None
            ):
                st.session_state.uploaded_name = None
                st.session_state.file_bytes = None
                st.session_state.df = None
                st.session_state.results = None
                st.session_state.pred_ts = None

    # Main header
    st.markdown(
        '<h1 class="main-header">🤖 ML Predictor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Transform your data into intelligent predictions with XGBoost Classifier</p>',
        unsafe_allow_html=True,
    )

    # Main content
    if st.session_state.df is not None:
        # Success banner
        _, c, _ = st.columns([1, 2, 1])
        with c:
            st.markdown(
                """
            <div class="success-message">
                ✅ <strong>File uploaded successfully!</strong><br>
                Ready for prediction analysis.
            </div>
            """,
                unsafe_allow_html=True,
            )

        # File info
        st.markdown("### 📊 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📝 Filename", st.session_state.uploaded_name)
        with c2:
            st.metric("📏 Rows", f"{len(st.session_state.df):,}")
        with c3:
            st.metric("📊 Columns", len(st.session_state.df.columns))
        with c4:
            size_kb = (len(st.session_state.file_bytes or b"")) / 1024
            st.metric("💾 Size", f"{size_kb:.1f} KB")

        # Preview
        st.markdown("### 👀 Data Preview")
        with st.expander("View your data", expanded=True):
            st.dataframe(
                st.session_state.df.head(100), use_container_width=True, height=300
            )

        # Stats
        if st.checkbox("📈 Show Data Statistics"):
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)

        # Predict button
        st.markdown("### 🔮 Make Predictions")
        b1, b2, b3 = st.columns([1, 1, 1])
        with b2:
            if st.button(
                "🚀 Generate Predictions", use_container_width=True, type="primary"
            ):
                if not st.session_state.file_bytes:
                    st.error("No file loaded.")
                else:
                    with st.spinner(
                        "🤖 Sending data to backend and generating predictions..."
                    ):
                        start = time.time()
                        data = send_to_api(
                            st.session_state.file_bytes,
                            st.session_state.uploaded_name,
                            api_url,
                        )
                        if (
                            data
                            and isinstance(data, dict)
                            and data.get("status") == "success"
                        ):
                            d = data.get("data", {})
                            preds = d.get("predictions", [])
                            proc = d.get("processing_time_seconds", None)
                            # Keep only as many predictions as rows
                            if len(preds) != len(st.session_state.df):
                                st.warning(
                                    f"Prediction count ({len(preds)}) does not match rows ({len(st.session_state.df)}). "
                                    "Truncating to min length."
                                )
                            n = min(len(preds), len(st.session_state.df))
                            st.session_state.results = {
                                "predictions": preds[:n],
                                "processing_time": (
                                    proc
                                    if proc is not None
                                    else round(time.time() - start, 3)
                                ),
                                "num_predictions": d.get("num_predictions", n),
                                "message": data.get("message", "Predictions generated"),
                            }
                            # set the timestamp now that we have results (filename stable across reruns)
                            if st.session_state.pred_ts is None:
                                st.session_state.pred_ts = int(time.time())
                        else:
                            st.session_state.results = None

        # Results
        if st.session_state.results:
            st.markdown("### 🎯 Prediction Results")
            r = st.session_state.results

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("✅ Status", "Success")
            with c2:
                st.metric("⚡ Processing Time", f"{r.get('processing_time', 0):.3f}s")
            with c3:
                st.metric(
                    "📊 Predictions Made",
                    r.get("num_predictions", len(r["predictions"])),
                )

            # Merge predictions with original DF (trim to n)
            n = len(r["predictions"])
            out_df = st.session_state.df.head(n).copy()
            out_df["Prediction"] = r["predictions"]

            st.markdown("### 📋 Detailed Predictions")
            st.dataframe(out_df, use_container_width=True, height=400)

            # Ensure pred_ts exists (should be set when predictions were generated)
            if st.session_state.pred_ts is None:
                st.session_state.pred_ts = int(time.time())

            fmt = st.selectbox(
                "Select download format", ("Parquet", "CSV"), key="download_format"
            )

            filename = f"predictions_{st.session_state['pred_ts']}.{ 'csv' if fmt == 'CSV' else 'parquet' }"

            # prepare bytes but catch any errors so we don't crash the rerun
            data_bytes = None
            mime = "application/octet-stream"
            error_msg = None

            try:
                if fmt == "CSV":
                    data_bytes = out_df.to_csv(index=False).encode("utf-8")
                    mime = "text/csv"
                else:
                    buf = io.BytesIO()
                    out_df.to_parquet(
                        buf, index=False
                    )  # pyarrow / fastparquet required
                    buf.seek(0)
                    data_bytes = buf.getvalue()
                    mime = "application/x-parquet"
            except Exception as e:
                error_msg = str(e)

            # layout + consistent widget rendering (download_button always present)
            _, mid, _ = st.columns([1, 1, 1])
            with mid:
                st.download_button(
                    label="📥 Download Predictions",
                    data=(
                        data_bytes if data_bytes is not None else b""
                    ),  # keep widget present
                    file_name=filename,
                    mime=mime,
                    key=f"download_{fmt.lower()}",  # stable per-format key
                    disabled=(data_bytes is None),
                    use_container_width=True,
                )

                # show helpful error if file creation failed
                if error_msg:
                    st.error(f"Could not create file: {error_msg}")

                # lightweight console log (appears in the iframe console)
                components.html(
                    f"<script>console.log({json.dumps({'event':'download_section_rendered','format':fmt,'error':bool(error_msg)})})</script>",
                    height=0,
                )

    else:
        # Empty state
        st.markdown(
            """
        <div class="file-upload-container">
            <h2>📁 Get Started</h2>
            <p>Upload your CSV or Excel file using the sidebar to begin making predictions!</p>
            <p>👈 Look for the file uploader in the sidebar</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
