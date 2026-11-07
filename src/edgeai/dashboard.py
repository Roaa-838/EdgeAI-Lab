"""Read-only dashboard: refuses to plot synthetic results as accuracy."""
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title='EdgeAI-Lab | Deployment trade-offs',layout='wide')
st.title('EdgeAI-Lab — Vision Deployment Benchmarks')
st.caption('Measurements must come from a real trained model, labeled held-out data, and matching checkpoints.')
file=Path('results/benchmark.csv')
if not file.exists():
    st.info('No real benchmark is present. Run `edgeai benchmark` first. Synthetic smoke tests are not model accuracy results.')
    st.stop()
df=pd.read_csv(file)
if df.empty or not (df['run_type']=='real_labeled_evaluation').all():
    st.error('The benchmark is not marked as real labeled evaluation.');st.stop()
selected=st.selectbox('Batch size',sorted(df['batch_size'].unique().tolist()))
cpu=df[(df.batch_size==selected)&(df.device=='cpu')]
st.subheader('Measured CPU inference results')
st.dataframe(cpu[['backend','top1_accuracy','top5_accuracy','median_latency_ms','p95_latency_ms','model_file_mb','peak_process_rss_mb']],hide_index=True,use_container_width=True)
left,right=st.columns(2)
with left:
    st.caption('Latency, ms (lower better)')
    st.bar_chart(cpu.set_index('backend')['median_latency_ms'])
with right:
    st.caption('Top-1 accuracy on selected validation images (higher better)')
    st.bar_chart(cpu.set_index('backend')['top1_accuracy'])
st.warning('Peak RSS is whole-process memory, including runtime dependencies; not isolated model RAM or GPU VRAM.')
st.caption('INT8 acceleration depends on CPU and ONNX Runtime operator support. A smaller file does not guarantee faster inference.')
