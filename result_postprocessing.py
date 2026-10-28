import pandas as pd
import numpy as np
from pathlib import Path

result_folder = "./results/nr_objects"

metric_name = "evaluation/joint/nmi"
for csv_path in sorted(Path(result_folder).glob("*.csv")):
    df = pd.read_csv(csv_path, sep = ",")
    df = df[df["State"]=="finished"]
    nmi_results = pd.to_numeric(df[metric_name], errors="coerce").values
    print(f"NMI mean {np.nanmean(nmi_results)}, std {np.nanstd(nmi_results)} ")