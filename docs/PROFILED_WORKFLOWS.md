# Profiled Workflow Set

These workflows mirror the DAG shapes used by the external SMIless workloads
while using different workflow and stage names. They are designed for
CPU/memory-backed mock execution instead of `sleep_ms`.

## civic_alert_flow

SMIless shape source: WL1 AMBER Alert.

```text
detect_object -> estimate_pose -> match_face -> classify_scene -> translate_alert
detect_object ---------------------------------> classify_scene
```

Config: `configs/civic_alert_flow.yaml`

## visual_qa_flow

SMIless shape source: WL2 Image-Query.

```text
image_embed -> text_embed -> answer_question
```

Config: `configs/visual_qa_flow.yaml`

## spoken_dialog_flow

SMIless shape source: WL3 Voice Assistant.

```text
speech_decode -> topic_route ----\
                                  -> response_generate -> speech_synthesize
speech_decode -> entity_extract -/
```

Config: `configs/spoken_dialog_flow.yaml`

## Execution Model

Each node specifies:

- `cpu_iters`: deterministic CPU loop work.
- `memory_kb`, `memory_passes`, `memory_stride`: memory allocation/touch work.
- `warm_overhead_ms`, `cold_overhead_ms`: offline trace simulation overhead priors.

Real OpenWhisk replay uses `actions/sebs_mock.py`, which now reads the CPU and
memory profile fields. Offline trace generation uses:

```bash
python -m runner.stage2_forecastor.simulate_profiled_stage_trace --help
```

The current reference run uses `civic_alert_flow` with the Azure periodic-drift
arrival schedule and `SLO=4000 ms`.
