"""Simulated Sentinel-2 surface reflectance for Overberg land-cover classes."""
import numpy as np
import pandas as pd

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# mean reflectance (x1e-4) per class; invasive trees share similar spectra -> realistic confusion
CLASSES = {
    "Pine (invasive)":       [250, 420, 260, 800, 2100, 2500, 2650, 2800, 1250, 560],
    "Eucalyptus (invasive)": [280, 470, 300, 880, 2300, 2750, 2900, 3050, 1400, 640],
    "Acacia (invasive)":     [300, 520, 320, 950, 2500, 3000, 3150, 3300, 1500, 700],
    "Fynbos (native)":       [450, 650, 700, 1150, 1700, 1900, 2050, 2150, 2300, 1500],
    "Agriculture":           [500, 800, 650, 1300, 2600, 3100, 3300, 3400, 2100, 1200],
    "Water":                 [700, 600, 400, 300, 220, 200, 180, 160, 80, 50],
    "Bare / urban":          [1300, 1500, 1700, 1850, 2000, 2100, 2200, 2300, 2900, 2500],
}
COLORS = {"Pine (invasive)": "#1B5E20", "Eucalyptus (invasive)": "#7CB342", "Acacia (invasive)": "#F9A825",
          "Fynbos (native)": "#8D6E63", "Agriculture": "#AED581", "Water": "#1E88E5", "Bare / urban": "#BDBDBD"}


def spectra(labels, rng):
    m = np.array([CLASSES[c] for c in labels], dtype=float)
    X = m * rng.lognormal(0, .09, m.shape) * rng.lognormal(0, .06, (len(labels), 1))
    df = pd.DataFrame(X.round(), columns=BANDS)
    df["NDVI"] = ((df.B8 - df.B4) / (df.B8 + df.B4)).round(4)
    return df


def training_samples(n_per_class=500, seed=51):
    rng = np.random.default_rng(seed)
    labels = np.repeat(list(CLASSES), n_per_class)
    df = spectra(labels, rng)
    df["label"] = labels
    return df


def scene(size=120, seed=52):
    """Spatially coherent class map (smoothed random fields) + its simulated spectra."""
    rng = np.random.default_rng(seed)
    names = list(CLASSES)
    fields = []
    for _ in names:
        f = rng.normal(size=(size, size))
        for _ in range(6):  # cheap smoothing
            f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
        fields.append(f)
    w = np.array([1.0, .9, 1.0, 1.25, 1.1, .7, .8])[:, None, None]
    truth = np.argmax(np.stack(fields) * w, axis=0)
    labels = np.array(names)[truth.ravel()]
    return truth, spectra(labels, rng)
