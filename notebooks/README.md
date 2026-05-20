# Azure Trace Notebooks

## Notebook

- `azure_trace_exploration.ipynb`

This notebook analyzes `AzureFunctionsInvocationTrace2021` and exports replay schedules for the SeBS-style OpenWhisk workflows.

## Expected Dataset

Use the Azure Functions invocation-level trace:

- `AzureFunctionsInvocationTrace2021`
- Schema: `app`, `func`, `end_timestamp`, `duration`
- `start_timestamp = end_timestamp - duration`

Azure is used only as the workflow entry arrival source. It is not a DAG trace.

## How to Run

```bash
# 1) Enter the prototype directory.
# Purpose: keep notebook paths relative to stage2-openwhisk-prototype.
cd ~/myproject/stage2-openwhisk-prototype

# 2) Point the notebook to the extracted Azure 2021 trace directory.
# Purpose: avoid hard-coding a machine-specific data path in the notebook.
export AZURE_TRACE_ROOT=/path/to/AzureFunctionsInvocationTrace2021

# 3) Start Jupyter.
# Purpose: open notebooks/azure_trace_exploration.ipynb and run cells step by step.
jupyter notebook
```

## Outputs

The notebook writes analysis outputs under:

```text
data/azure_analysis/
```

Important files:

- `azure2021_basic_function_summary.csv`
- `azure2021_candidate_minute_counts.csv`
- `azure2021_candidate_characterization.csv`
- `schedule_<label>_<workflow>.csv`

## Selection Goal

Pick representative `(app, func)` traces for:

- sparse workloads
- bursty workloads
- periodic workloads
- mixed / drift workloads

Then replay those schedules onto:

- `sebs_trip_booking`
- `sebs_video`
- `sebs_map_reduce`
- `sebs_ml`
