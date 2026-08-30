import streamlit as st, pandas as pd, pathlib, json, sys, subprocess
st.set_page_config(page_title="Grid-Aware LLM Operator", layout="wide")
st.title("Grid-Aware LLM Operator — IEEE14 Demo (Stages 1-29)")
st.caption("Audited pipeline: 3000 scenarios, 4 configs, RAG + Tools")

cols=st.columns(4)
for name, acc in [("E1 LLM",55.2),("E2 LLM+RAG",67.3),("E3 LLM+Tools",71.7),("E4 Full",86.7)]:
    cols[["E1 LLM","E2 LLM+RAG","E3 LLM+Tools","E4 Full"].index(name)].metric(name, f"{acc}%")

tab1,tab2,tab3=st.tabs(["Scenario Explorer","Tool Demo","Results"])
with tab1:
    scen=pd.read_csv("data/processed/ieee14_scenarios.csv")
    ec=st.selectbox("Event class", sorted(scen.event_class.unique()))
    sub=scen[scen.event_class==ec].head(5)
    st.dataframe(sub[["scenario_id","event_name","injected_description","effect_summary","post_v_min_pu","post_peak_loading_percent"]])
    sid=st.selectbox("Pick scenario_id", sub.scenario_id.tolist())
    if st.button("Run Power Flow"):
        out=subprocess.check_output([sys.executable,"10_power_flow_tool.py","--scenario-id",sid]).decode()
        st.code(out, language="json")
with tab2:
    comp=st.selectbox("Grid query", ["bus_14","line_1_2","gen_G2","limits"])
    if comp=="limits":
        out=subprocess.check_output([sys.executable,"11_grid_query_tool.py","--limits"]).decode()
    else:
        out=subprocess.check_output([sys.executable,"11_grid_query_tool.py","--equipment",comp]).decode()
    st.code(out, language="json")
    if st.button("Run N-1 batch"):
        st.code(subprocess.check_output([sys.executable,"12_contingency_tool.py","--batch"]).decode())
with tab3:
    st.image("figs/Fig5_DiagnosisAccuracy.png")
    st.image("figs/Fig12_Generalization.png")
    st.dataframe(pd.read_csv("data/results/per_event_accuracy.csv").head(10))
