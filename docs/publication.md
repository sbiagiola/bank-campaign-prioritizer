# Public portfolio scope

## Recommended boundary for this project

Publish a complete, reproducible benchmark implementation. The source code demonstrates modeling and engineering decisions; hiding the core training or evaluation code would make the results harder to assess.

| Include publicly | Keep local or private |
| --- | --- |
| Source code and configuration with benchmark settings | Credentials, environment files, and Streamlit secrets |
| Tests and dependency lockfile | Virtual environments, caches, and editor state |
| English documentation and explicit limitations | Agent settings and local working notes |
| Aggregate results and diagnostic plots | Generated model binaries, which can be retrained |
| Benchmark labels and model scores used by the demo, with UCI attribution | Downloaded raw archives and duplicate dataset files |
| A downloader and commands to reproduce the experiment | Any future client data, private business rules, or client-specific integrations |

The `.gitignore` implements the local-file boundary for this benchmark. It does not change remote visibility or remove anything already present in Git history.

## Why the full implementation helps

For an ML portfolio, readers should be able to inspect how data leakage is addressed, how preprocessing is fitted, which data influences model selection, and how results are computed. The value of this project is the reasoning and reproducibility around a standard public benchmark.

For client work, a separate private repository can hold client datasets, connectors, credentials, margins, and business logic. Public case studies should contain only material the client has authorized for publication. This benchmark repository need not include those materials to demonstrate the underlying approach.

## Dataset and source-code licensing

The [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing) lists Bank Marketing under CC BY 4.0. Attribution appears in the README; the raw dataset is downloaded separately. The prediction export is derived from the public benchmark and retains that attribution.

Public visibility and a source-code license are separate choices. This repository does not yet specify a code license. Select one before presenting it as licensed open-source software; the dataset's license does not license the project's Python code.

## Publication status

This document describes the intended public scope. It does not mean the repository has been pushed, made public, deployed, or connected to LinkedIn. The README reports offline results and the app presents historical scenarios.
