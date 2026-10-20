"""
Pure Python/NumPy re-implementation of the pykep 2.x routines used by the
official ESA "Trappist Tour" evaluation code.

Why this file exists
--------------------
The official evaluation code (official/trappist_tour_official.py) depends on
pykep and pygmo, which are only distributed as compiled binaries (conda-forge).
To keep the project self-contained and easy to read, every astrodynamics routine
the fitness function needs is re-implemented here, following the pykep 2.6
C++ sources as closely as possible (same formulas, same branches, same
iteration tolerances), so that the numbers agree with pykep to ~1e-9 relative.

Implemented routines (names mirror pykep):
    par2ic, ic2par, keplerian planet ephemerides (Kepler equation),
    lambert_problem (Izzo 2015, zero revolutions),
    propagate_lagrangian (Lagrange coefficients, Kepler/hyperbolic Kepler eq.),
    fb_prop (unpowered gravity-assist).

Only scalars and 3-vectors are used: the routines are called a few dozen
times per fitness evaluation, so plain `math` is faster than NumPy here.
"""
import math

# ---------------------------------------------------------------- constants
# identical to pykep/core/constants.h
AU = 149597870691.0          # [m]
DAY2SEC = 86400.0
DEG2RAD = math.pi / 180.0
EARTH_RADIUS = 6378137.0     # [m]

_PI = math.pi


# ---------------------------------------------------------------- 3-vectors
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


# ---------------------------------------------------------------- elements
def par2ic(E, mu):
    """Osculating elements [a, e, i, W, w, EA] -> Cartesian (r, v).

    EA is the eccentric anomaly (ellipse) as in pykep::par2ic."""
    a, e, i, W, w, EA = E
    b = a * math.sqrt(1.0 - e * e)
    n = math.sqrt(mu / (a * a * a))
    # perifocal frame
    cE, sE = math.cos(EA), math.sin(EA)
    xper = a * (cE - e)
    yper = b * sE
    xdotper = -(a * n * sE) / (1.0 - e * cE)
    ydotper = (b * n * cE) / (1.0 - e * cE)
    # rotation perifocal -> inertial
    cW, sW = math.cos(W), math.sin(W)
    cw, sw = math.cos(w), math.sin(w)
    ci, si = math.cos(i), math.sin(i)
    R00 = cW * cw - sW * sw * ci
    R01 = -cW * sw - sW * cw * ci
    R10 = sW * cw + cW * sw * ci
    R11 = -sW * sw + cW * cw * ci
    R20 = sw * si
    R21 = cw * si
    r = (R00 * xper + R01 * yper, R10 * xper + R11 * yper, R20 * xper + R21 * yper)
    v = (R00 * xdotper + R01 * ydotper, R10 * xdotper + R11 * ydotper,
         R20 * xdotper + R21 * ydotper)
    return r, v


def _acos(x):
    # C++ std::acos returns NaN outside [-1, 1]; Python raises. Mirror C++.
    if -1.0 <= x <= 1.0:
        return math.acos(x)
    return float("nan")


def _sqrt(x):
    return math.sqrt(x) if x >= 0.0 else float("nan")


def ic2par(r, v, mu):
    """Cartesian -> [a, e, i, W, w, EA]  (pykep::ic2par).

    EA is the eccentric anomaly for e < 1 and the hyperbolic (Gudermannian)
    anomaly for e > 1, both in (-pi, pi]. Only a, e and EA are needed by the
    Trappist fitness, the angles are computed for completeness."""
    h = _cross(r, v)
    Hn = _norm(h)
    n = (-h[1], h[0], 0.0)          # node line  k x h
    Nn = _norm(n)
    R = _norm(r)
    vxh = _cross(v, h)
    ev = (vxh[0] / mu - r[0] / R, vxh[1] / mu - r[1] / R, vxh[2] / mu - r[2] / R)
    p = Hn * Hn / mu
    e = _norm(ev)
    a = p / (1.0 - e * e)
    i = _acos(h[2] / Hn)
    if Nn > 0.0:
        W = _acos(n[0] / Nn)
        if n[1] < 0.0:
            W = 2.0 * _PI - W
        w = _acos(_dot(n, ev) / Nn / e) if e > 0 else float("nan")
        if ev[2] < 0.0:
            w = 2.0 * _PI - w
    else:
        W = w = float("nan")
    ni = _acos(_dot(ev, r) / e / R)            # true anomaly
    if _dot(r, v) < 0.0:
        ni = 2.0 * _PI - ni
    if e < 1.0:
        EA = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(ni / 2.0))
    else:
        arg = math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(ni / 2.0)
        EA = math.atanh(arg) if -1.0 < arg < 1.0 else float("nan")
    return a, e, i, W, w, EA


def mean_to_eccentric(M, e):
    """Kepler equation M = E - e sin E (Newton, pykep::m2e)."""
    E = M + e * math.cos(M)
    for _ in range(100):
        f = E - e * math.sin(E) - M
        E_new = E - f / (1.0 - e * math.cos(E))
        if abs(E_new - E) < 1e-15:
            return E_new
        E = E_new
    return E


class Planet:
    """Keplerian planet (pykep.planet.keplerian)."""

    def __init__(self, ref_mjd2000, elements, mu_central, mu_self, radius,
                 safe_radius, name):
        self.ref_mjd2000 = ref_mjd2000
        self.elements = tuple(float(x) for x in elements)   # a,e,i,W,w,M
        self.mu_central = mu_central
        self.mu_self = mu_self
        self.radius = radius
        self.safe_radius = safe_radius
        self.name = name
        a = self.elements[0]
        self.mean_motion = math.sqrt(mu_central / (a * a * a))
        self.period_days = 2 * math.pi / self.mean_motion / DAY2SEC

    def eph(self, mjd2000):
        a, e, i, W, w, M0 = self.elements
        dt = (mjd2000 - self.ref_mjd2000) * DAY2SEC
        M = M0 + self.mean_motion * dt
        EA = mean_to_eccentric(M, e)
        return par2ic((a, e, i, W, w, EA), self.mu_central)


# ---------------------------------------------------------------- Lambert
def _hypergeometric_F(z, tol):
    Sj, Cj, err, j = 1.0, 1.0, 1.0, 0
    while err > tol:
        Cj1 = Cj * (3.0 + j) * (1.0 + j) / (2.5 + j) * z / (j + 1)
        Sj += Cj1
        err = abs(Cj1)
        Cj = Cj1
        j += 1
    return Sj


def _x2tof2(x, lam):
    a = 1.0 / (1.0 - x * x)
    if a > 0:  # ellipse
        alfa = 2.0 * math.acos(x)
        beta = 2.0 * math.asin(math.sqrt(lam * lam / a))
        if lam < 0.0:
            beta = -beta
        return a * math.sqrt(a) * ((alfa - math.sin(alfa)) - (beta - math.sin(beta))) / 2.0
    alfa = 2.0 * math.acosh(x)
    beta = 2.0 * math.asinh(math.sqrt(-lam * lam / a))
    if lam < 0.0:
        beta = -beta
    return -a * math.sqrt(-a) * ((beta - math.sinh(beta)) - (alfa - math.sinh(alfa))) / 2.0


def _x2tof(x, lam):
    """Non-dimensional time of flight T(x) for N = 0 (Izzo 2015)."""
    battin, lagrange = 0.01, 0.2
    dist = abs(x - 1.0)
    if battin < dist < lagrange:
        return _x2tof2(x, lam)
    K = lam * lam
    E = x * x - 1.0
    rho = abs(E)
    z = math.sqrt(1.0 + K * E)
    if dist < battin:          # Battin series
        eta = z - lam * x
        S1 = 0.5 * (1.0 - lam - x * eta)
        Q = 4.0 / 3.0 * _hypergeometric_F(S1, 1e-11)
        return (eta ** 3 * Q + 4.0 * lam * eta) / 2.0
    y = math.sqrt(rho)         # Lancaster
    g = x * z - lam * E
    if E < 0:
        d = math.acos(g)
    else:
        d = math.log(y * (z - lam * x) + g)
    return (x - lam * z - d / y) / E


def _dTdx(x, T, lam):
    l2 = lam * lam
    l3 = l2 * lam
    umx2 = 1.0 - x * x
    y = math.sqrt(1.0 - l2 * umx2)
    y2 = y * y
    y3 = y2 * y
    DT = 1.0 / umx2 * (3.0 * T * x - 2.0 + 2.0 * l3 * x / y)
    DDT = 1.0 / umx2 * (3.0 * T + 5.0 * x * DT + 2.0 * (1.0 - l2) * l3 / y3)
    DDDT = 1.0 / umx2 * (7.0 * x * DDT + 8.0 * DT - 6.0 * (1.0 - l2) * l2 * l3 * x / y3 / y2)
    return DT, DDT, DDDT


def lambert(r1, r2, tof, mu, cw=False):
    """Zero-revolution Lambert problem, Izzo's algorithm (pykep::lambert_problem).

    Returns (v1, v2). Returns NaN velocities if the iteration breaks down, which
    the fitness function then handles with the official death penalty."""
    if tof <= 0:
        raise ValueError("Time of flight is negative!")
    c = (r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2])
    cn = _norm(c)
    R1 = _norm(r1)
    R2 = _norm(r2)
    s = (cn + R1 + R2) / 2.0
    ir1 = (r1[0] / R1, r1[1] / R1, r1[2] / R1)
    ir2 = (r2[0] / R2, r2[1] / R2, r2[2] / R2)
    ih = _cross(ir1, ir2)
    hn = _norm(ih)
    ih = (ih[0] / hn, ih[1] / hn, ih[2] / hn)
    lambda2 = 1.0 - cn / s
    lam = math.sqrt(lambda2)
    if ih[2] < 0.0:            # transfer angle larger than pi
        lam = -lam
        it1 = _cross(ir1, ih)
        it2 = _cross(ir2, ih)
    else:
        it1 = _cross(ih, ir1)
        it2 = _cross(ih, ir2)
    n1, n2 = _norm(it1), _norm(it2)
    it1 = (it1[0] / n1, it1[1] / n1, it1[2] / n1)
    it2 = (it2[0] / n2, it2[1] / n2, it2[2] / n2)
    if cw:
        lam = -lam
        it1 = (-it1[0], -it1[1], -it1[2])
        it2 = (-it2[0], -it2[1], -it2[2])
    lambda3 = lam * lambda2
    T = math.sqrt(2.0 * mu / s ** 3) * tof

    # initial guess (single revolution branch only)
    T00 = math.acos(lam) + lam * math.sqrt(1.0 - lambda2)
    T1 = 2.0 / 3.0 * (1.0 - lambda3)
    if T >= T00:
        x = -(T - T00) / (T - T00 + 4.0)
    elif T <= T1:
        x = T1 * (T1 - T) / (2.0 / 5.0 * (1.0 - lambda2 * lambda3) * T) + 1.0
    else:
        x = (T / T00) ** (0.69314718055994529 / math.log(T1 / T00)) - 1.0

    # Householder iterations (3rd order), same tolerance as pykep
    try:
        err, it = 1.0, 0
        while err > 1e-5 and it < 15:
            tof_x = _x2tof(x, lam)
            DT, DDT, DDDT = _dTdx(x, tof_x, lam)
            delta = tof_x - T
            DT2 = DT * DT
            xnew = x - delta * (DT2 - delta * DDT / 2.0) / (DT * (DT2 - delta * DDT) + DDDT * delta * delta / 6.0)
            err = abs(x - xnew)
            x = xnew
            it += 1
    except (ValueError, ZeroDivisionError, OverflowError):
        nan = float("nan")
        return (nan, nan, nan), (nan, nan, nan)

    gamma = math.sqrt(mu * s / 2.0)
    rho = (R1 - R2) / cn
    sigma = math.sqrt(1.0 - rho * rho)
    y = math.sqrt(1.0 - lambda2 + lambda2 * x * x)
    vr1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / R1
    vr2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / R2
    vt = gamma * sigma * (y + lam * x)
    vt1, vt2 = vt / R1, vt / R2
    v1 = (vr1 * ir1[0] + vt1 * it1[0], vr1 * ir1[1] + vt1 * it1[1], vr1 * ir1[2] + vt1 * it1[2])
    v2 = (vr2 * ir2[0] + vt2 * it2[0], vr2 * ir2[1] + vt2 * it2[1], vr2 * ir2[2] + vt2 * it2[2])
    return v1, v2


# ---------------------------------------------------------------- propagation
def propagate_lagrangian(r0, v0, t, mu):
    """Keplerian propagation with Lagrange coefficients (pykep::propagate_lagrangian).

    Raises RuntimeError when Kepler's equation cannot be solved, exactly like
    pykep (the official code catches it in _check_distance)."""
    R = _norm(r0)
    V = _norm(v0)
    energy = V * V / 2.0 - mu / R
    a = -mu / 2.0 / energy
    sigma0 = _dot(r0, v0) / math.sqrt(mu)

    if a > 0:  # ellipse: solve for the eccentric anomaly difference DE
        sqrta = math.sqrt(a)
        DM = math.sqrt(mu / a ** 3) * t
        DE = DM
        k1 = sigma0 / sqrta
        k2 = 1.0 - R / a
        # f(DE) is strictly increasing (f' = r/a > 0) and changes sign in
        # [DM - pi, DM + pi] (same bracket as pykep). Halley steps safeguarded
        # by bisection -> always converges to the unique root.
        lo, hi = DM - _PI, DM + _PI
        for _ in range(200):
            sE, cE = math.sin(DE), math.cos(DE)
            f = DE - DM + k1 * (1.0 - cE) - k2 * sE
            if f > 0:
                hi = DE
            else:
                lo = DE
            df = 1.0 + k1 * sE - k2 * cE
            d2f = k1 * cE + k2 * sE
            den = 2 * df * df - f * d2f
            DE_new = DE - 2 * f * df / den if den != 0 else 0.5 * (lo + hi)
            if not (lo < DE_new < hi):
                DE_new = 0.5 * (lo + hi)
            if abs(DE_new - DE) <= 4e-16 * max(1.0, abs(DE_new)) or hi - lo < 1e-15:
                DE = DE_new
                break
            DE = DE_new
        else:
            raise RuntimeError("Maximum number of iterations exceeded (Kepler equation)")
        cE, sE = math.cos(DE), math.sin(DE)
        r = a + (R - a) * cE + sigma0 * sqrta * sE
        F = 1.0 - a / R * (1.0 - cE)
        G = a * sigma0 / math.sqrt(mu) * (1.0 - cE) + R * math.sqrt(a / mu) * sE
        Ft = -math.sqrt(mu * a) / (r * R) * sE
        Gt = 1.0 - a / r * (1.0 - cE)
    else:      # hyperbola: hyperbolic anomaly difference DH
        sqrta = math.sqrt(-a)
        DN = math.sqrt(-mu / a ** 3) * t
        k1 = sigma0 / sqrta
        k2 = 1.0 - R / a
        # f(DH) = -DH - DN + k1 (cosh DH - 1) + k2 sinh DH is strictly increasing
        # (f' = r/|a| > 0): bracket the unique root, then safeguarded Newton.
        def fH(DH):
            return -DH - DN + k1 * (math.cosh(DH) - 1.0) + k2 * math.sinh(DH)
        try:
            lo, hi = -1.0, 1.0
            while fH(lo) > 0:
                lo *= 2.0
            while fH(hi) < 0:
                hi *= 2.0
        except OverflowError:
            raise RuntimeError("Hyperbolic Kepler equation diverged")
        DH = 0.5 * (lo + hi)
        for _ in range(300):
            sH, cH = math.sinh(DH), math.cosh(DH)
            f = -DH - DN + k1 * (cH - 1.0) + k2 * sH
            if f > 0:
                hi = DH
            else:
                lo = DH
            df = -1.0 + k1 * sH + k2 * cH
            DH_new = DH - f / df if df > 0 else 0.5 * (lo + hi)
            if not (lo < DH_new < hi):
                DH_new = 0.5 * (lo + hi)
            if abs(DH_new - DH) <= 4e-16 * max(1.0, abs(DH_new)) or hi - lo < 1e-15 * max(1.0, abs(DH)):
                DH = DH_new
                break
            DH = DH_new
        else:
            raise RuntimeError("Maximum number of iterations exceeded (hyperbolic Kepler)")
        cH, sH = math.cosh(DH), math.sinh(DH)
        r = a + (R - a) * cH + sigma0 * sqrta * sH
        F = 1.0 - a / R * (1.0 - cH)
        G = a * sigma0 / math.sqrt(mu) * (1.0 - cH) + R * math.sqrt(-a / mu) * sH
        Ft = -math.sqrt(-mu * a) / (r * R) * sH
        Gt = 1.0 - a / r * (1.0 - cH)

    rn = (F * r0[0] + G * v0[0], F * r0[1] + G * v0[1], F * r0[2] + G * v0[2])
    vn = (Ft * r0[0] + Gt * v0[0], Ft * r0[1] + Gt * v0[1], Ft * r0[2] + Gt * v0[2])
    return rn, vn


# ---------------------------------------------------------------- fly-by
def fb_prop(v_in, v_pla, rp, beta, mu):
    """Unpowered fly-by: outgoing velocity (pykep::fb_prop)."""
    vrel = (v_in[0] - v_pla[0], v_in[1] - v_pla[1], v_in[2] - v_pla[2])
    vrel2 = _dot(vrel, vrel)
    vreln = math.sqrt(vrel2)
    ecc = 1.0 + rp / mu * vrel2
    delta = 2.0 * math.asin(1.0 / ecc)
    i_hat = (vrel[0] / vreln, vrel[1] / vreln, vrel[2] / vreln)
    j_hat = _cross(i_hat, v_pla)
    jn = _norm(j_hat)
    j_hat = (j_hat[0] / jn, j_hat[1] / jn, j_hat[2] / jn)
    k_hat = _cross(i_hat, j_hat)
    c1 = vreln * math.cos(delta)
    c2 = vreln * math.cos(beta) * math.sin(delta)
    c3 = vreln * math.sin(beta) * math.sin(delta)
    return (v_pla[0] + c1 * i_hat[0] + c2 * j_hat[0] + c3 * k_hat[0],
            v_pla[1] + c1 * i_hat[1] + c2 * j_hat[1] + c3 * k_hat[1],
            v_pla[2] + c1 * i_hat[2] + c2 * j_hat[2] + c3 * k_hat[2])
