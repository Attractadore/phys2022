#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

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

data_26 = np.array((
    0.07, 0,
    1.73, 1.22,
    1.85, 4.27,
    1.90, 6.55,
    2.09, 17.1,
    2.13, 19.2,
    2.21, 21.5,
    2.31, 22.4,
    2.41, 22.0,
    2.51, 21.2,
    2.62, 20.1,
    2.73, 18.9,
    2.80, 18.1,
    2.91, 17.1,
    3.03, 16.0,
    3.13, 15.3,
    3.24, 14.7,
    3.34, 14.3,
    3.44, 13.8,
    3.52, 13.6,
    3.62, 13.1,
    3.74, 12.8,
    3.80, 12.6,
    3.92, 12.5,
    4.03, 12.2,
    4.12, 12.1,
    4.30, 11.8,
    4.61, 11.4,
    4.94, 11.0,
    5.23, 10.7,
    5.51, 10.5,
    5.82, 10.3,
    6.11, 10.2,
    6.41, 10.0,
    6.52, 9.95,
    6.65, 9.92,
    6.84, 9.72,
    7.01, 9.70,
    7.12, 9.75,
    7.22, 9.67,
    7.32, 9.70,
    7.44, 9.73,
    7.54, 9.77,
    7.84, 9.90,
    8.13, 10.1,
))

data_24 = np.array((
    0.82, 0,
    0.90, 0,
    1.52, 0,
    1.73, 0.4,
    1.90, 2.8,
    2.00, 5.4,
    2.14, 8.1,
    2.25, 8.8,
    2.25, 8.9,
    2.30, 9.0,
    2.35, 8.8,
    2.44, 8.6,
    2.63, 7.7,
    2.86, 6.5,
    3.25, 5.4,
    3.53, 5.0,
    3.85, 4.7,
    4.26, 4.4,
    4.70, 4.2,
    5.09, 4.0,
    5.30, 3.9,
    5.72, 3.7,
    6.27, 3.6,
    6.43, 3.6,
    6.57, 3.6,
    6.65, 3.6,
    6.72, 3.5,
    6.72, 3.6,
    6.80, 3.7,
    6.92, 3.7,
    7.01, 3.7,
    7.55, 3.7,
    7.69, 3.7,
    8.05, 3.8,
    8.60, 3.9,
))

# data = np.array([
#     0.20, 251,
#     0.40, 242,
#     0.60, 257,
#     0.80, 287,
#     1.00, 522,
#     1.20, 812,
#     1.40, 992,
#     1.60, 1040,
#     1.80, 1059,
#     2.00, 969,
#     2.20, 865,
#     2.40, 674,
#     2.60, 500,
#     2.80, 408,
#     3.00, 471,
#     3.20, 770,
#     3.40, 232,
#     3.60, 155,
#     3.80, 294,
#     4.00, 299,
#     4.20, 247,
#     4.40, 135,
#     4.60, 123,
#     3.30, 348,
#     3.25, 498,
#     3.10, 652,
#     3.15, 727,
#     3.23, 600,
#     3.70, 328,
#     4.30, 170,
#     1.70, 931,
#     3.65, 179,
#     3.50, 194,
#     3.75, 250,
# ])
# 
# Idata_base = data[::2]
# idx = np.argsort(Idata_base)
# Idata_base = Idata_base[idx]
# Ndata_base = data[1::2][idx]
# 
# write_string("data.tab.tex", "\n".join((
# r"""\begin{tabular}{
#     S[
#         table-alignment-mode = none,
#         round-mode = places,
#         round-precision = 2,
#     ]
#     |
#     S[
#         table-alignment-mode = none,
#         round-mode = places,
#         round-precision = 0,
#     ]
# }
# \toprule
# $I \text{, } \unit{A}$ & $N \text{, } \unit{1/s}$ \\
# \midrule""",
# 
# "\n".join(f"{I} & {N} \\\\" for (I, N) in zip(
#     Idata_base, Ndata_base
# )),
# 
# r"""\bottomrule
# \end{tabular}""",
# )))
# 
# noise = 260
# noise_err = 16
# Ndata_base -= noise
# 
# Idata = Idata_base
# Ndata = Ndata_base
# 
# conv_i = np.where(Idata == 3.20)[0][0]
# Iconv = Idata[conv_i]
# Nconv = Ndata[conv_i]
# Tconv = 662 * 1000
# pconv = p_from_T(Tconv)
# k = pconv / Iconv
# 
# write_string("Tconv.tex",
#     "\n".join((
# r"""$""",
# r"""T_c = \qty[
#     round-mode = figures,
#     round-precision = 3,
# ]{""", "{0}".format(Tconv / 1000), r"""}{keV} """,
# r"""$""",
# )))
# 
# write_string("noise.tex",
#     "\n".join((
# r"""$""",
# r"""N_n = \qty[
#     uncertainty-mode = separate,
#     round-mode = uncertainty,
#     round-precision = 1,
# ]{""", "{0} +- {1}".format(noise, noise_err), r"""}{1/s} """,
# r"""$""",
# )))
# 
# write_string("conv.tex",
#     "\n".join((
# r"""\begin{gather*}""",
# r"""T_c = \qty[
#     round-mode = figures,
#     round-precision = 3,
# ]{""", "{0}".format(Tconv / 1000), r"""}{keV} \\ """,
# r"""p_c = \qty[
#     round-mode = figures,
#     round-precision = 3,
# ]{""", "{0}".format(pconv * 1000), r"""}{\frac{meV}{c}} \\ """,
# r"""I_c = \qty[
#     round-mode = places,
#     round-precision = 2,
# ]{""", "{0}".format(Iconv), r"""}{A} \\ """,
# r"""k = \frac{p_c}{I_c} = \qty[
#     round-mode = figures,
#     round-precision = 3,
# ]{""", "{0}".format(k * 1000), r"""}{\frac{meV}{c \cdot A}} \\ """,
# r"""\end{gather*}""",
# )))
# 
# nidx = np.where(Ndata >= 0.0)
# Idata = Idata[nidx]
# Ndata = Ndata[nidx]
# 
# pdata = k * Idata
# Tdata = T_from_p(pdata)
# 
# pdata *= 1000
# Tdata /= 1000
# 
# fidx = np.where(np.logical_and(Tdata >= 240, Tdata <= 600))
# fdata = (np.sqrt(Ndata / pdata) / pdata)
# 
# popt, pcov = np.polyfit(fdata[fidx], Tdata[fidx], deg=1, cov=True)
# popt = popt[::-1]
# pcov = np.fliplr(np.transpose(np.fliplr(pcov)))
# T_max = popt[0]
# T_max_err = np.sqrt(pcov[0, 0])
# alpha = popt[1]
# 
# f = np.polynomial.Polynomial(popt)
# errp = np.zeros(2 * popt.size - 1)
# for i in range(popt.size):
#     errp[2 * i] += pcov[i, i]
#     for j in range(i + 1, popt.size):
#         errp[i + j] += 2 * pcov[i, j]
# errp = np.polynomial.Polynomial(errp)
# 
# def errf(n, err = 0):
#     coef_err2 = errp(n)
#     n_err2 = (f.deriv()(n) * err) ** 2
#     return np.sqrt(coef_err2 + n_err2)
# 
# estI = np.linspace(0, p_from_T(T_max * 1000) / k, Idata_base.size)
# estp = k * estI
# estT = T_from_p(estp)
# estp *= 1000
# estT /= 1000
# estN = (estp ** 3) / (alpha ** 2) * (estT - T_max) ** 2
# 
# plt.plot(Idata_base, Ndata_base, 'o', label=r"Спектр $\beta$-частиц")
# plt.plot(estI, estN, '--', label="Аппроксимация по графику Ферми")
# plt.title(r"Измерение спектра $\beta$-частиц")
# plt.xlabel(r"$I, \unit{\ampere}$")
# plt.ylabel(r"$N, \unit{1/\second}$")
# plt.grid()
# plt.legend()
# plt.savefig("data.pgf")
# 
# write_string("Tmax.tex",
# "\n".join((
# r"""\begin{gather*}""",
# r"""T = T_{max} - \alpha \frac{\sqrt{N}}{p^{3/2}} \\ """,
# r"""T_{max} = \qty[
#     uncertainty-mode = separate,
#     round-mode = uncertainty,
#     round-precision = 1,
# ]{""", f"{T_max} +- {T_max_err}", r"""}{keV} """,
# r"""\end{gather*}""",
# )))
# 
# plt.figure(clear=True)
# plt.plot(pdata, Ndata, 'o')
# plt.title(r"Измерение спектра $\beta$-частиц")
# plt.xlabel(r"$p, \unit{meV/c}$")
# plt.ylabel(r"$N, \unit{1/s}$")
# plt.grid()
# plt.savefig("data2.pgf")
# 
# plt.figure(clear=True)
# plt.plot(Tdata, Ndata, 'o')
# plt.title(r"Измерение спектра $\beta$-частиц")
# plt.xlabel(r"$T, \unit{keV}$")
# plt.ylabel(r"$N, \unit{1/s}$")
# plt.grid()
# plt.savefig("data3.pgf")
# 
# estx = np.array((0, np.max(fdata[fidx])))
# esty = f(estx)
# 
# plt.figure(clear=True)
# plt.plot(Tdata, fdata, 'o', label=r"Спектр $\beta$-частиц")
# plt.plot(esty, estx, '--', label="Аппроксимация")
# plt.title(r"График Ферми")
# plt.xlabel(r"$T, \unit{keV}$")
# plt.ylabel(r"$\frac{\sqrt{N}}{p^{3/2}}, \unit{\frac{1}{\sqrt{s}}.\frac{c^{3/2}}{{meV}^{3/2}}}$")
# plt.grid()
# plt.legend()
# plt.savefig("fermi.pgf")
# 
# write_string("data2.tab.tex", "\n".join((
# r"""\begin{tabular}{
#     S[
#         table-alignment-mode = none,
#         round-mode = places,
#         round-precision = 2,
#     ]
#     |
#     S[
#         table-alignment-mode = none,
#         round-mode = places,
#         round-precision = 0,
#     ]
#     |
#     S[
#         table-alignment-mode = none,
#         round-mode = figures,
#         round-precision = 3,
#     ]
#     |
#     S[
#         table-alignment-mode = none,
#         round-mode = figures,
#         round-precision = 3,
#     ]
#     |
#     S[
#         table-alignment-mode = none,
#         round-mode = figures,
#         round-precision = 3,
#     ]
# }
# \toprule
# $I \text{, } \unit{A}$ & $N \text{, } \unit{1/s}$ & $T \text{, } \unit{keV}$ & $p \text{, } \unit{meV/c}$ & $\frac{\sqrt{N}}{p^{3/2}} \text{, } \unit{\frac{1}{\sqrt{s}}.\frac{c^{3/2}}{{meV}^{3/2}}}$ \\
# \midrule""",
# 
# "\n".join(f"{I} & {N} & {T} & {p} & {y} \\\\" for (I, N, T, p, y) in zip(
#     Idata, Ndata, Tdata, pdata, fdata,
# )),
# 
# r"""\bottomrule
# \end{tabular}""",
# )))
