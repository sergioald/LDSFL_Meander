
from __future__ import annotations

import numpy as np


def resistance_function_flagbed(flagbed: int, theta0: float, ds: float, rpic_0: float):
    """
    Port of Resistance_Function_flagbed.m

    Notes:
      The original MATLAB file does not explicitly set rpic/F0 in the flagbed==1 branch.
      For safety we set:
        rpic = rpic_0
        F0 = sqrt(1.65*ds*theta0/Cf0)
    """
    flagbed = int(flagbed)
    theta0 = float(theta0)
    ds = float(ds)
    rpic_0 = float(rpic_0)

    if flagbed == 1:
        XO = 1.0/(6.0 + 2.5*np.log(1.0/(2.5*ds)))
        Cf0 = XO**2
        CD = -5.0*XO
        CT = 0.0

        thetacr = 0.0495
        dth = theta0 - thetacr
            


        if dth >= 0.0:
            phiT = 1.5*theta0*(dth**(-1.0))
        else:
            # MATLAB-style: allow complex instead of NaN
            dthc = complex(dth, 0.0)
            phiT = 1.5*theta0*(dthc**(-1.0))
            raise ValueError(f"theta0 <= thetacr (theta0={theta0}, thetacr={thetacr})")

        phiD = 0.0

        rpic = rpic_0
        F0 = np.sqrt(1.65*ds*theta0/Cf0)
        return rpic, float(Cf0), float(CT), float(CD), phiT, float(phiD), float(F0)

    if flagbed == 2:
        Teta0 = theta0
        Teta1 = 0.06 + 0.3*Teta0**1.5  # Engelund-Fredsoe 1972
        if (Teta1 > 0.55) or (Teta1 < 0.0474):
            Teta1 = Teta0
        AA = Teta1/Teta0
        DTeta1 = 0.45*np.sqrt(Teta0)

        XO = 1.0/(6.0 + 2.5*np.log(AA/(2.5*ds)))
        BB = 1.0/Teta0 - DTeta1/Teta1
        XX = XO*XO
        Cf0 = XX/AA
        CD = -5.0*XO
        CT = (1.0 + 5.0*XO)*BB*Teta0

        phiD = -CD
        phiT = 2.5 - CT

        rpic = rpic_0/np.sqrt(AA)
        F0 = np.sqrt(1.65*ds*theta0/Cf0)
        return float(rpic), float(Cf0), float(CT), float(CD), float(phiT), float(phiD), float(F0)

    raise ValueError(f"Unsupported flagbed={flagbed}")
