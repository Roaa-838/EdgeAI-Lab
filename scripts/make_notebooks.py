import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'notebooks'
p.mkdir(parents=True,exist_ok=True)

def nb(cells):
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':v.splitlines(keepends=True)} if kind=='markdown' else
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':v.splitlines(keepends=True)}
        for kind,v in cells]}
files={
 '01_dataset_and_preprocessing.ipynb':[
   ('markdown','# Dataset integrity and preprocessing\nRun after downloading Imagenette. Training/calibration live in train; held-out evaluation lives in val.'),
   ('code','from pathlib import Path\nfrom edgeai.dataset import classes_in_directory, list_images, stratified_limit, preprocess\nroot = Path("../data/imagenette2-160")\nclasses = classes_in_directory(root / "train")\nprint(classes)\n'),
   ('code','mapping={name:i for i,name in enumerate(classes)}\ntrain=list_images(root/"train",mapping)\nval=list_images(root/"val",mapping)\nprint(f"train={len(train)}, held-out val={len(val)}")\nprint("Preprocessed input:", preprocess(val[0]["path"]).shape)\n'),
 ],
 '02_experiment_results.ipynb':[
   ('markdown','# Deployment trade-off analysis\nOnly run after `edgeai benchmark`. Plot measured values; do not treat smoke-test data as research outcomes.'),
   ('code','import pandas as pd\nfrom pathlib import Path\np = Path("../results/benchmark.csv")\nassert p.is_file(), "Run actual model benchmark first"\ndf = pd.read_csv(p)\nassert df.run_type.eq("real_labeled_evaluation").all()\ndf[["backend","device","batch_size","top1_accuracy","median_latency_ms","p95_latency_ms","model_file_mb"]]\n'),
   ('code','import matplotlib.pyplot as plt\ncpu=df[(df.device=="cpu") & (df.batch_size==1)]\nax=cpu.plot.bar(x="backend",y="median_latency_ms",legend=False,figsize=(8,4))\nax.set_ylabel("Measured CPU latency (ms)")\nplt.tight_layout()\n'),
 ]}
for filename,cells in files.items():
    (p/filename).write_text(json.dumps(nb(cells),indent=2),encoding='utf-8')
