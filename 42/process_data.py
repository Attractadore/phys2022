#!/usr/bin/env python3
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import electron_mass, speed_of_light

def p_from_T(T):
    """
    T^2 + 2Tmc^2 = p^2c^2
    p = sqrt((T/c)^2 + 2Tm)
    """
    return np.sqrt((T/speed_of_light)**2 + 2*T*electron_mass)

def T_from_p(p):
    """
    T = sqrt(p^2c^2 + m^2c^4) - mc^2
    """
    idle = electron_mass * speed_of_light ** 2
    return np.sqrt((p*speed_of_light)**2 + idle ** 2) - idle

def write_string(name, s):
    with open(name, "w") as f:
        f.write(s)

plt.rcParams.update({
    "pgf.preamble": r"""
        \usepackage[main=russian, english]{babel}
        \usepackage[utf8]{inputenc}
        \usepackage{siunitx}
    """,
    "text.usetex": True,
    "font.family": "serif",
    "font.monospace": [],
    "font.sans-serif": [],
    "font.serif": [],
})
plt.rcParams["text.latex.preamble"] = plt.rcParams["pgf.preamble"]

data = np.array([
    0.20, 251,
    0.40, 242,
    0.60, 257,
    0.80, 287,
    1.00, 522,
    1.20, 812,
    1.40, 992,
    1.60, 1040,
    1.80, 1059,
    2.00, 969,
    2.20, 865,
    2.40, 674,
    2.60, 500,
    2.80, 408,
    3.00, 471,
    3.20, 770,
    3.40, 232,
    3.60, 155,
    3.80, 294,
    4.00, 299,
    4.20, 247,
    4.40, 135,
    4.60, 123,
    3.30, 348,
    3.25, 498,
    3.10, 652,
    3.15, 727,
    3.23, 600,
    3.70, 328,
    4.30, 170,
    1.70, 931,
    3.65, 179,
    3.50, 194,
    3.75, 250,
])

Idata_base = data[::2]
idx = np.argsort(Idata_base)
Idata_base = Idata_base[idx]
Ndata_base = data[1::2][idx]

write_string("data.tab.tex", "\n".join((
r"""\begin{tabular}{
    S[
        table-alignment-mode = none,
        round-mode = places,
        round-precision = 2,
    ]
    |
    S[
        table-alignment-mode = none,
        round-mode = places,
        round-precision = 0,
    ]
}
\toprule
$I \text{, } \unit{A}$ & $N \text{, } \unit{1/s}$ \\
\midrule""",

"\n".join(f"{I} & {N} \\\\" for (I, N) in zip(
    Idata_base, Ndata_base
)),

r"""\bottomrule
\end{tabular}""",
)))

noise = 260
noise_err = 16
Ndata_base -= noise

Idata = Idata_base
Ndata = Ndata_base

conv_i = np.where(Idata == 3.20)[0][0]
Iconv = Idata[conv_i]
Nconv = Ndata[conv_i]
Tconv = 662 * 1000
pconv = p_from_T(Tconv)
k = pconv / Iconv

write_string("Tconv.tex",
    "\n".join((
r"""$""",
r"""T_c = \qty[
    round-mode = figures,
    round-precision = 3,
]{""", "{0}".format(Tconv / 1000), r"""}{keV} """,
r"""$""",
)))

write_string("noise.tex",
    "\n".join((
r"""$""",
r"""N_n = \qty[
    uncertainty-mode = separate,
    round-mode = uncertainty,
    round-precision = 1,
]{""", "{0} +- {1}".format(noise, noise_err), r"""}{1/s} """,
r"""$""",
)))

write_string("conv.tex",
    "\n".join((
r"""\begin{gather*}""",
r"""T_c = \qty[
    round-mode = figures,
    round-precision = 3,
]{""", "{0}".format(Tconv / 1000), r"""}{keV} \\ """,
r"""p_c = \qty[
    round-mode = figures,
    round-precision = 3,
]{""", "{0}".format(pconv * 1000), r"""}{\frac{meV}{c}} \\ """,
r"""I_c = \qty[
    round-mode = places,
    round-precision = 2,
]{""", "{0}".format(Iconv), r"""}{A} \\ """,
r"""k = \frac{p_c}{I_c} = \qty[
    round-mode = figures,
    round-precision = 3,
]{""", "{0}".format(k * 1000), r"""}{\frac{meV}{c \cdot A}} \\ """,
r"""\end{gather*}""",
)))

nidx = np.where(Ndata >= 0.0)
Idata = Idata[nidx]
Ndata = Ndata[nidx]

pdata = k * Idata
Tdata = T_from_p(pdata)

pdata *= 1000
Tdata /= 1000

fidx = np.where(np.logical_and(Tdata >= 240, Tdata <= 600))
fdata = (np.sqrt(Ndata / pdata) / pdata)

popt, pcov = np.polyfit(fdata[fidx], Tdata[fidx], deg=1, cov=True)
popt = popt[::-1]
pcov = np.fliplr(np.transpose(np.fliplr(pcov)))
T_max = popt[0]
T_max_err = np.sqrt(pcov[0, 0])
alpha = popt[1]

f = np.polynomial.Polynomial(popt)
errp = np.zeros(2 * popt.size - 1)
for i in range(popt.size):
    errp[2 * i] += pcov[i, i]
    for j in range(i + 1, popt.size):
        errp[i + j] += 2 * pcov[i, j]
errp = np.polynomial.Polynomial(errp)

def errf(n, err = 0):
    coef_err2 = errp(n)
    n_err2 = (f.deriv()(n) * err) ** 2
    return np.sqrt(coef_err2 + n_err2)

estI = np.linspace(0, p_from_T(T_max * 1000) / k, Idata_base.size)
estp = k * estI
estT = T_from_p(estp)
estp *= 1000
estT /= 1000
estN = (estp ** 3) / (alpha ** 2) * (estT - T_max) ** 2

plt.plot(Idata_base, Ndata_base, 'o', label=r"Спектр $\beta$-частиц")
plt.plot(estI, estN, '--', label="Аппроксимация по графику Ферми")
plt.title(r"Измерение спектра $\beta$-частиц")
plt.xlabel(r"$I, \unit{\ampere}$")
plt.ylabel(r"$N, \unit{1/\second}$")
plt.grid()
plt.legend()
plt.savefig("data.pgf")

write_string("Tmax.tex",
"\n".join((
r"""\begin{gather*}""",
r"""T = T_{max} - \alpha \frac{\sqrt{N}}{p^{3/2}} \\ """,
r"""T_{max} = \qty[
    uncertainty-mode = separate,
    round-mode = uncertainty,
    round-precision = 1,
]{""", f"{T_max} +- {T_max_err}", r"""}{keV} """,
r"""\end{gather*}""",
)))

plt.figure(clear=True)
plt.plot(pdata, Ndata, 'o')
plt.title(r"Измерение спектра $\beta$-частиц")
plt.xlabel(r"$p, \unit{meV/c}$")
plt.ylabel(r"$N, \unit{1/s}$")
plt.grid()
plt.savefig("data2.pgf")

plt.figure(clear=True)
plt.plot(Tdata, Ndata, 'o')
plt.title(r"Измерение спектра $\beta$-частиц")
plt.xlabel(r"$T, \unit{keV}$")
plt.ylabel(r"$N, \unit{1/s}$")
plt.grid()
plt.savefig("data3.pgf")

estx = np.array((0, np.max(fdata[fidx])))
esty = f(estx)

plt.figure(clear=True)
plt.plot(Tdata, fdata, 'o', label=r"Спектр $\beta$-частиц")
plt.plot(esty, estx, '--', label="Аппроксимация")
plt.title(r"График Ферми")
plt.xlabel(r"$T, \unit{keV}$")
plt.ylabel(r"$\frac{\sqrt{N}}{p^{3/2}}, \unit{\frac{1}{\sqrt{s}}.\frac{c^{3/2}}{{meV}^{3/2}}}$")
plt.grid()
plt.legend()
plt.savefig("fermi.pgf")

write_string("data2.tab.tex", "\n".join((
r"""\begin{tabular}{
    S[
        table-alignment-mode = none,
        round-mode = places,
        round-precision = 2,
    ]
    |
    S[
        table-alignment-mode = none,
        round-mode = places,
        round-precision = 0,
    ]
    |
    S[
        table-alignment-mode = none,
        round-mode = figures,
        round-precision = 3,
    ]
    |
    S[
        table-alignment-mode = none,
        round-mode = figures,
        round-precision = 3,
    ]
    |
    S[
        table-alignment-mode = none,
        round-mode = figures,
        round-precision = 3,
    ]
}
\toprule
$I \text{, } \unit{A}$ & $N \text{, } \unit{1/s}$ & $T \text{, } \unit{keV}$ & $p \text{, } \unit{meV/c}$ & $\frac{\sqrt{N}}{p^{3/2}} \text{, } \unit{\frac{1}{\sqrt{s}}.\frac{c^{3/2}}{{meV}^{3/2}}}$ \\
\midrule""",

"\n".join(f"{I} & {N} & {T} & {p} & {y} \\\\" for (I, N, T, p, y) in zip(
    Idata, Ndata, Tdata, pdata, fdata,
)),

r"""\bottomrule
\end{tabular}""",
)))
