# PyMC-Marketing vs. Meridian
We compare the two libraries in the following categories, from most important to least:
- Contribution recovery
- Predictive power
- Sampling efficiency (ESS / s)
- RAM footprint in sampling and of the final fitted model

## CPU tests

When working with GCP instances with preinstalled Jupyter Lab,
we have found Conda to be the easiest method.

```shell
conda env create -f cpu-environment.yaml
```
Open [comparison-all-cpu-adstock.ipynb] and run it with the newly created Conda environment named **python312-cpu**.

## GPU tests

```shell
conda create -n python311 python=3.11 -y && conda activate python311
```

```shell
conda install pip
```

```shell
pip install --user -r gpu-requirements-compiled.txt
```

```shell
ipython kernel install --name python311 --display-name "Python 3.11"  --user
```
