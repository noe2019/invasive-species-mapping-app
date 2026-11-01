# Invasive Alien Tree Mapping – Overberg (Sentinel-2)

Supervised classification of Sentinel-2 spectral bands to map invasive alien trees (pine, eucalyptus, acacia) in the Overberg region, South Africa.

**Live app:** https://noe-invasive-species-814a7fc4c770.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- Spectral signatures of 7 land-cover classes (10 bands + NDVI)
- Random Forest vs. Gradient Boosting comparison (accuracy, kappa)
- Confusion matrix
- Classified map and invaded area (ha)

## Tech stack
Python, Sentinel-2, scikit-learn, Streamlit, Plotly

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
