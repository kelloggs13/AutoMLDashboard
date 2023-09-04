# AutoML Dashboard

This repository contains two streamlit dashboards for fitting and scoring classification models.

Both dashboards are hosted on streamlit.app:

-   [Fitting a model](https://automl-fitting.streamlit.app/)
-   [Scoring new data](https://automl-scoring.streamlit.app/)

### Code Structure

Both dashboards have separate scripts for their respective streamlit components, but share a common script which includes custom functions as well as all the imported packages:

-   run_app_fitting.py \> app_fitting.py + functions.py + import_packages.py

-   run_app_scoring.py \> app_scoring.py + functions.py + import_packages.py

This setup allows to re-use custom functions (most notably for pre-processing the model features) as well as hosting them as separate dashboards on streamlit.app
