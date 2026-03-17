# DOES: Dynamics of Opinion and Emotion in Social Media

A Python implementation of the model presented in the paper "Modeling the Dynamics of Opinion and Emotion on Social Media," published in *IEEE Transactions on Computational Social Systems* (2025).

## Overview

The DOES model simulates how opinions and emotions co-evolve in social networks through two coupled dynamic layers.

## Installation

```bash
git clone https://github.com/WenyingG05/DOES.git
cd DOES
pip install -r requirements.txt
```

## Quick Start

Run the example:

```bash
python main.py
```

This will:
1. Generate a Barabási-Albert network with 100 nodes
2. Initialize random opinions (0-1) and emotion phases (-π to π)
3. Run the coupled dynamics simulation for 20 time units
4. Save a visualization to `DOES_example.png`

## Code Structure

```
DOES/
├── DOES_model.py     # Core model implementation
├── main.py           # Simple example and visualization
├── README.md         # This file
└── requirements.txt  # Dependencies
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{gong2025modeling,
	title={Modeling the Dynamics of Opinion and Emotion on Social Media},
	author={Gong, Wenying and Ye, Dongsheng and Li, Hao and Lin, Hui and Jiang, Hao},
	journal={IEEE Transactions on Computational Social Systems},
	year={2025},
	publisher={IEEE}
}
```
