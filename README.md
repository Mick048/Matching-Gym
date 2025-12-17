# Matching-Gym (Dynamic Matching + Static LP Solver)

This repo contains a simple dynamic matching simulation (Gym-style environment + agents) and a static LP solver used inside policies.

## 0. Clone the Repositary

```bash
git clone git@github.com:Mick048/Matching-Gym.git
```

## 1. Create + Activate env

```bash
conda env create -f environment.yml
conda activate dynamic-matching
```

## 2. Verify installation

```bash
python -c "import numpy, scipy, pulp, gymnasium; print('imports ok')"
which cbc
cbc -stop
```

## 3. Quick Run

```bash
python main.py
```
