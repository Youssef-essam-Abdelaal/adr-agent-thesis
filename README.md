# adr-agent-thesis

ADR-Agent (Adaptive Diagnosis and Recovery) is a training-free wrapper around an existing GUI agent. It works in five
stages: Prevent, Verify, Diagnose, Recover and Check the fix. The Monitor does Prevent and Verify and checks every fix,
the Diagnostician names the type of a flagged failure with the failure taxonomy (taxonomy v4), and the Strategist
chooses and runs the fix for that type. The agent itself is not changed. This repository holds the code of the
master's thesis on ADR-Agent.

Main files: `adr/wrapper.py` (the loop and the experiment conditions), `adr/monitor.py`, `adr/diagnostician.py` and
`adr/strategist.py` (the Monitor, the Diagnostician and the Strategist of ADR v1 and v2), `adr/v3/` (ADR v3),
`adr/env/` (MiniWoB++ with fault injection, and OSWorld), `adr/config.py` (every setting) and
`taxonomy/taxonomy_v4.json` (the taxonomy the Diagnostician prompt is built from).

## Install

Needs Python 3.9 or newer and Google Chrome (Playwright drives the installed Chrome).

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    git clone https://github.com/Farama-Foundation/miniwob-plusplus.git third_party/miniwob-plusplus
    git -C third_party/miniwob-plusplus checkout 33c3b4ddef8c6eb67c57a29663d844b1eda7e614

The OpenAI key is read from `OPENAI_API_KEY` or `~/.openai_key`.

## Tests

The tests need no key, browser or model:

    ADR_NO_OPENAI=1 ADR_FAKE_LLM=1 .venv/bin/python -m unittest discover -s tests

`ADR_FAKE_LLM=1` replaces every model call with a fake reply, so it also gives a dry run of a whole experiment.
`ADR_NO_OPENAI=1` makes any call that would reach OpenAI fail.

## MiniWoB++ experiment

    .venv/bin/python -m adr.experiment --name try --tasks login-user,click-checkboxes --seeds 1 \
        --faults none,grounding --conditions base,adr

Each episode is saved to `runs/try__<fingerprint>/` (the runner prints the folder), and the same command resumes a
stopped run. The settings are in `adr/config.py`; change them with `--set key=value` or a JSON file passed with
`--config`. The default is ADR v2. The final settings, ADR v3.7, are ADR v3 with these switches:

    V37="--set adr_version=3 --set click_check=v5 --set retype=1 --set date_context=1 --set done_always=1
      --set no_effect_max_frac=0.1 --set n_free=1 --set drop_blocked=1 --set diag_rules=1 --set fix_report=1
      --set ground_fields=1 --set max_recoveries=10 --set replan_gap=3 --set done_pairs=1 --set use_axes=1
      --set replan_strikes=2 --set gate_dismiss_first=1"
    .venv/bin/python -m adr.experiment --name try37 --tasks login-user --seeds 1 --conditions base,adr $V37

Numbers and a results page (`results/report.html`):

    .venv/bin/python -m adr.analyze --runs try__<fingerprint> --rq1 ""

## OSWorld experiment

OSWorld is not included. Clone https://github.com/xlang-ai/OSWorld into `third_party/OSWorld`, set up its VMware
virtual machine as its README describes (`vmrun` must be on the PATH), install this repository's requirements into
OSWorld's environment, and run from there with one worker (`$V37` as above):

    third_party/OSWorld/.venv/bin/python -m adr.experiment --name osw --config configs/osworld_open.json $V37 \
        --tasks <task ids> --seeds 1 --faults none --conditions base,adr --workers 1

Task ids are the names of the JSON files (without `.json`) in
`third_party/OSWorld/evaluation_examples/examples/chrome/`; other apps are chosen with `--set osworld_domain=...`.
`configs/osworld_open.json` (15 steps) expects UI-TARS-1.5-7B served by vLLM at `127.0.0.1:18000` (model name
`ui-tars`) and Qwen3-VL-32B (4-bit) in Ollama at `127.0.0.1:11435` as judge, Diagnostician and element finder; change
the `*_base_url` and `*_model` keys for another setup. `configs/osworld_e25.json` uses GPT-4o as judge and
Diagnostician, and 50 steps.

## Data for the offline evaluations

The datasets are not included.

- CUAErrorBench, for `adr/rq1_offline.py`, `adr/rq1_agentic.py` and `adr/v3/replay.py`:
  https://huggingface.co/datasets/CyT1ng/cua_debugger_traj. Put its `data/train.parquet` into `data/cuaerrorbench/`,
  with the Claude Sonnet 4.5 archive (144 runs) as `claude.zip` and the Gemini 2.5 Pro archive (30 runs) as
  `gemini.zip`. `data/cuaerrorbench/tasks.json` maps `<domain>/<task id>` to the task's instruction, taken from
  OSWorld's `evaluation_examples/examples/<domain>/<task id>.json`.
- Judge benchmarks, for `adr/judge_bench.py`: GUI-Reflection, GUI-Critic-Test and CUAStepBench from Hugging Face,
  under `data/e23/`. The docstring of `adr/judge_bench.py` gives the dataset names and the layout.
- The 34 labeled OSWorld cases (`adr/rq1_offline.py --dataset osworld34`) are not public.

## Credits

`adr/agent_uitars.py` uses the computer-use prompt of UI-TARS (https://github.com/bytedance/UI-TARS), in the form
used by the official OSWorld runs of UI-TARS-1.5-7B. `adr/judge_bench.py` uses the action-verification prompt of
GUI-Reflection (https://github.com/penghao-wu/GUI_Reflection) for its comparison runs.

## License

MIT, see `LICENSE`.
