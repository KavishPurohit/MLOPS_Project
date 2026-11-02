from pathlib import Path
import sys, json, os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RAW_DIR, FEATURES
from src.data import load_fd001, add_training_rul
from src.features import engineer_features

st.set_page_config(page_title="AeroPulse | Engine Health", page_icon="✈️", layout="wide")
st.markdown("<style>.stApp{background:#f5f7fb}.block-container{padding-top:1.7rem}.hero{background:linear-gradient(120deg,#10233f,#146a75);padding:1.5rem 2rem;border-radius:18px;color:white;margin-bottom:1rem}.hero h1{color:#fff;margin:0}</style>", unsafe_allow_html=True)
st.sidebar.title("✈️ AeroPulse")
st.sidebar.caption("Predictive maintenance · NASA C-MAPSS")
page = st.sidebar.radio("Navigate", ["Project Overview", "Dataset Explorer", "Exploratory Data Analysis", "ETL & Feature Store", "Model Experiments", "Model Evaluation", "RUL Prediction", "Monitoring", "MLOps Pipeline"])
st.markdown('<div class="hero"><h1>Aircraft Engine Health Intelligence</h1><p>Remaining useful life estimates from sequential sensor history · FD001 starter configuration</p></div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_data(): return load_fd001(RAW_DIR)

def dataset_or_message():
    try: return get_data()
    except Exception as e:
        st.info(f"FD001 data is not available yet: {e}")
        st.markdown("Download and extract NASA C-MAPSS into `data/raw/`. See README for exact steps.")
        return None

if page == "Project Overview":
    st.subheader("From sensor history to maintenance planning")
    st.write("A connected workflow ingests run-to-failure histories, constructs engine-aware features, compares regressors, serves estimates with FastAPI, and exposes operational telemetry.")
    a,b,c=st.columns(3); a.metric("Starter subset","FD001"); b.metric("Input signals","3 settings + 21 sensors"); c.metric("Prediction unit","one engine history")
    st.graphviz_chart('digraph {rankdir=LR; NASA->ETL->FeatureStore->Training->Model; Model->FastAPI->Streamlit; FastAPI->Prometheus->Grafana; GitHubActions->Tests; Terraform->Docker; Kubernetes->API}')
    st.caption("Model quality values appear after a training run against official labels.")
elif page == "Dataset Explorer":
    pack=dataset_or_message()
    if pack:
        tr,te,labels=pack; st.write(f"Training rows: {len(tr):,} · training engines: {tr.engine_id.nunique()} · test engines: {te.engine_id.nunique()}")
        eng=st.selectbox("Training engine", sorted(tr.engine_id.unique())); view=tr[tr.engine_id==eng]
        signals=st.multiselect("Signals", FEATURES, default=["sensor_2","sensor_3"])
        if signals: st.plotly_chart(px.line(view,x="cycle",y=signals,title=f"Engine {eng} history"),use_container_width=True)
        st.dataframe(tr.head(30),use_container_width=True)
elif page == "Exploratory Data Analysis":
    pack=dataset_or_message()
    if pack:
        tr,_,_=pack; sig=st.selectbox("Sensor", FEATURES[3:],index=1)
        st.plotly_chart(px.line(tr.groupby("cycle")[sig].mean().reset_index(),x="cycle",y=sig,title="Mean sensor trajectory across training engines"),use_container_width=True)
        st.plotly_chart(px.imshow(tr[FEATURES].corr(),title="Input feature correlations",aspect="auto"),use_container_width=True)
elif page == "ETL & Feature Store":
    pack=dataset_or_message()
    st.write("Validation: 26 whitespace-delimited columns, finite numeric values, positive cycles, sorted engine/cycle keys. Training RUL is last observed cycle minus current cycle; test RUL uses NASA's one label per engine plus remaining observed cycles.")
    st.write("Features: current value, life-to-date mean/std, rolling means/stds (5, 10, 20 cycles), 10-cycle slope proxy, elapsed cycle count and log cycle count. All features use only current and past data.")
    if pack:
        tr,_,_=pack; f=engineer_features(add_training_rul(tr).groupby("engine_id").head(2)); st.dataframe(f.head(),use_container_width=True)
        st.caption("Feature definitions are shared by training and inference through src/features.py.")
elif page == "Model Experiments":
    p=ROOT/"models/evaluation.json"
    if p.exists():
        result=json.loads(p.read_text()); st.subheader("Validation comparison (engine-wise split)"); st.dataframe(pd.DataFrame(result["validation"]).T,use_container_width=True)
        st.write("MLflow UI: `mlflow ui --backend-store-uri ./mlruns` · http://127.0.0.1:5000")
    else: st.info("No experiment results yet. Add FD001 and run `python -m src.train` to produce MLflow runs.")
elif page == "Model Evaluation":
    p=ROOT/"models/evaluation.json"
    if p.exists():
        result=json.loads(p.read_text()); st.metric("Selected model",result["selected_model"]); st.json(result["test"])
    else: st.info("Evaluation requires a trained model and NASA test RUL labels. No metrics are fabricated.")
elif page == "RUL Prediction":
    source=st.radio("History source",["Upload CSV","Example test engine"],horizontal=True)
    uploaded=st.file_uploader("Upload an engine sensor-history CSV",type="csv") if source == "Upload CSV" else None
    engine_frame=None
    if source == "Example test engine":
        pack=dataset_or_message()
        if pack:
            _,test_df,_=pack
            chosen=st.selectbox("Test engine",sorted(test_df.engine_id.unique()))
            engine_frame=test_df[test_df.engine_id==chosen][["engine_id","cycle",*FEATURES]]
    api=st.text_input("Prediction API",os.getenv("API_URL","http://localhost:8000"))
    if (uploaded or engine_frame is not None) and st.button("Estimate remaining cycles"):
        content=uploaded.getvalue() if uploaded else engine_frame.to_csv(index=False).encode()
        filename=uploaded.name if uploaded else f"engine_{chosen}.csv"
        response=requests.post(f"{api}/predict",files={"file":(filename,content,"text/csv")},timeout=30)
        if response.ok: st.success(f"Estimated RUL: {response.json()['estimated_rul_cycles']:.1f} cycles")
        else: st.error(response.text)
    st.caption("CSV columns: cycle, op_setting_1..3, sensor_1..21; engine_id is optional.")
elif page == "Monitoring":
    endpoint=st.text_input("Prometheus API",os.getenv("PROMETHEUS_URL","http://localhost:9090"))
    for title,query in [("Prediction requests","rul_prediction_requests_total"),("Error requests","rul_prediction_requests_total{status=\"error\"}"),("Latency p95","histogram_quantile(0.95, sum(rate(rul_prediction_latency_seconds_bucket[5m])) by (le))")]:
        try:
            r=requests.get(f"{endpoint}/api/v1/query",params={"query":query},timeout=3).json()["data"]["result"]
            st.write(title,r[0]["value"][1] if r else "No samples scraped yet")
        except Exception: st.write(title,"Prometheus unavailable or no API metrics scraped")
    st.caption("Operational metrics describe service behavior. Model quality against labels is offline and shown in Model Evaluation.")
elif page == "MLOps Pipeline":
    st.subheader("Lifecycle runbook")
    st.markdown("1. Git: branch from main, commit focused changes, open a pull request.\n2. CI: GitHub Actions runs pytest on pushes and pull requests.\n3. Container: `docker compose up --build` runs API and dashboard.\n4. Kubernetes: `kubectl apply -f kubernetes/`; scale with `kubectl scale deployment/rul-api --replicas=5`.\n5. Terraform: use `terraform/` independently to manage one Docker API container.\n6. Monitoring: Prometheus scrapes `/metrics`; Grafana dashboard is provisioned from `monitoring/`.")
    st.write("Scrum artifacts and practical checklist are in `docs/`.")
