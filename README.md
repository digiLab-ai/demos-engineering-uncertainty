# Fusion Uncertainty Workshop

A small teaching repo demonstrating uncertainty propagation across three interconnected models:

1. Temperature model (heat flux -> peak temperature)
2. Irradiation model (neutron flux -> peak dpa)
3. Yield strength model (temperature & dpa -> yield strength)

The repo exposes a Streamlit app with three tabs, one per model. Each tab shows:
- A short model summary and placeholder for an explanatory graphic.
- Simple stochastic model behaviour (aleatoric uncertainty).
- Downloadable CSVs for training/validation inputs and outputs.
