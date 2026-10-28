"""load the odours"""

import numpy as np
from drosolf import pns

def load_odours():
    pn = pns.pns()
    odours = pn.to_numpy(dtype=float)
    odour_names = list(pn.index)

    return odours, odour_names

if __name__ == "__main__":
    odours, odour_names = load_odours()

    print("Shape:", odours.shape)
    print("Missing values:", np.isnan(odours).sum())
    print(f"Min: {odours.min():.2f} Max: {odours.max():.2f} Mean: {odours.mean():.2f}")
    print("First 5 odours:", odour_names[:5])

