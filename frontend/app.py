import time
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

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
        bio = BytesIO(file_bytes)
        if filename.lower().endswith(".csv"):
            return pd.read_csv(bio)
        elif filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(bio)
        elif filename.lower().endswith(".parquet"):
            return pd.read_parquet(bio)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
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
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = None  # streamlit UploadedFile
    if "file_bytes" not in st.session_state:
        st.session_state.file_bytes = None  # raw bytes (to send to API)
    if "df" not in st.session_state:
        st.session_state.df = None  # preview dataframe
    if "results" not in st.session_state:
        st.session_state.results = None  # API response (parsed)

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
        <h2>🤖 DataMinds'25 ML Predictor</h2>
        <p>Upload your data and get predictions from the backend model.</p>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        api_url = "http://127.0.0.1:8000/predict"

        st.markdown("### 📁 Upload Your Data")
        uploaded = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Supported: .csv, .xlsx, .xls, .parquet",
        )

        # Handle new/removed file
        if uploaded is not None:
            if uploaded is not st.session_state.uploaded:
                # Persist uploaded file + bytes
                st.session_state.uploaded = uploaded
                st.session_state.file_bytes = uploaded.getvalue()  # safe to reuse
                st.session_state.df = load_df_from_bytes(
                    st.session_state.file_bytes, uploaded.name
                )
                st.session_state.results = None
        else:
            # removed
            st.session_state.uploaded = None
            st.session_state.file_bytes = None
            st.session_state.df = None
            st.session_state.results = None

    # Main header
    st.markdown(
        '<h1 class="main-header">🤖 DataMinds\'25 ML Predictor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Transform your data into intelligent predictions with cutting-edge machine learning</p>',
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
            st.metric("📝 Filename", st.session_state.uploaded.name)
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
                            st.session_state.uploaded.name,
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
                        else:
                            st.session_state.results = None

        # Results
        if st.session_state.results:
            st.markdown("### 🎯 Prediction Results")
            r = st.session_state.results

            c1, c2, c3 = st.columns(3)
            with c1:
                # If your backend later returns accuracy, wire it here; for now show message
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

            # Download
            _, mid, _ = st.columns([1, 1, 1])
            with mid:
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=out_df.to_csv(index=False),
                    file_name=f"predictions_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
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

        st.markdown("### ✨ What makes this special?")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "**🚀 Lightning Fast**\n- Optimized ML algorithms\n- Real-time processing\n- Instant results"
            )
        with c2:
            st.markdown(
                "**🎯 High Accuracy**\n- State-of-the-art models\n- Validated predictions\n- Confidence scoring"
            )
        with c3:
            st.markdown(
                "**📊 Easy to Use**\n- Drag & drop interface\n- Automatic data validation\n- Export ready results"
            )

        st.markdown("### 📋 Expected Data Format")
        sample_data = pd.DataFrame(
            {
                "Feature_1": [1.2, 2.1, 3.4, 4.2],
                "Feature_2": [0.8, 1.5, 2.3, 1.7],
                "Feature_3": [10, 15, 20, 12],
                "Category": ["A", "B", "A", "C"],
            }
        )
        st.dataframe(sample_data, use_container_width=True)
        st.caption("💡 Your data should have features in columns and samples in rows")


if __name__ == "__main__":
    main()
