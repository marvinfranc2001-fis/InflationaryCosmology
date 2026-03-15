import numpy as np
from cobaya.theory import Theory

# ============================================================
# 1️⃣ STAROBINSKY SR1 (Slow-roll primer orden)
# ============================================================

class EffectiveStarobinsky_SR1(Theory):
    """
    Starobinsky inflation
    First-order slow-roll approximation
    """

    params = {"N_star": None}

    def calculate(self, state, want_derived=True, **params):

        N = params["N_star"]

        # Slow-roll parameters (leading order)
        epsilon = 3.0 / (4.0 * N**2)
        eta = -1.0 / N

        # Observables
        n_s = 1.0 - 6.0 * epsilon + 2.0 * eta
        r = 16.0 * epsilon

        state["params"]["n_s"] = n_s
        state["params"]["r"] = r

        return True


# ============================================================
# 2️⃣ STAROBINSKY SR2 (Slow-roll segundo orden + running)
# ============================================================

class EffectiveStarobinsky_SR2(Theory):
    """
    Starobinsky inflation
    Second-order slow-roll corrections
    Includes running alpha_s
    """

    params = {"N_star": None}

    def calculate(self, state, want_derived=True, **params):

        N = params["N_star"]

        # Slow-roll parameters
        epsilon = 3.0 / (4.0 * N**2)
        eta = -1.0 / N
        xi2 = 1.0 / (N**2)

        # Second-order spectral index
        n_s = (
            1.0
            - 6.0 * epsilon
            + 2.0 * eta
            - 24.0 * epsilon**2
            + 16.0 * epsilon * eta
            - 2.0 * xi2
        )

        # Tensor-to-scalar ratio (second-order correction)
        C = -0.73
        r = 16.0 * epsilon * (1.0 + C * (epsilon - eta))

        # Running
        alpha_s = -24.0 * epsilon**2 + 16.0 * epsilon * eta - 2.0 * xi2

        state["params"]["n_s"] = n_s
        state["params"]["r"] = r
        state["params"]["alpha_s"] = alpha_s

        return True


# ============================================================
# 3️⃣ STAROBINSKY SR2 + REHEATING
# ============================================================

class EffectiveStarobinsky_SR2_reh(Theory):
    """
    Starobinsky inflation
    Second-order slow-roll + running
    Reheating parameterized via log10_Treh
    """

    params = {"log10_Treh": None}

    def calculate(self, state, want_derived=True, **params):

        log10_Treh = params["log10_Treh"]
        Treh = 10.0 ** log10_Treh

        # --- Reheating correction to N ---
        # Simple physically motivated parametrization
        # assuming w_reh = 0
        delta_N = -0.25 * np.log(Treh / 1e15)

        N = 55.0 + delta_N

        # --- Slow-roll parameters ---
        epsilon = 3.0 / (4.0 * N**2)
        eta = -1.0 / N
        xi2 = 1.0 / (N**2)

        # --- Second-order spectral index ---
        n_s = (
            1.0
            - 6.0 * epsilon
            + 2.0 * eta
            - 24.0 * epsilon**2
            + 16.0 * epsilon * eta
            - 2.0 * xi2
        )

        # --- Tensor-to-scalar ratio ---
        C = -0.73
        r = 16.0 * epsilon * (1.0 + C * (epsilon - eta))

        # --- Running ---
        alpha_s = -24.0 * epsilon**2 + 16.0 * epsilon * eta - 2.0 * xi2

        state["params"]["n_s"] = n_s
        state["params"]["r"] = r
        state["params"]["alpha_s"] = alpha_s

        return True

