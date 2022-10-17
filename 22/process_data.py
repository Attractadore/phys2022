#!/usr/bin/env python3
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def write_string(name, s):
    with open(name, "w") as f:
        f.write(s)

n_err = 2

neon = np.loadtxt("neon.txt")
xneon = neon[:, 1]
yneon = neon[:, 0]

mercury = np.loadtxt("mercury.txt")
xmercury = mercury[:, 1]
ymercury = mercury[:, 0]

xdata = np.concatenate((xneon, xmercury))
ydata = np.concatenate((yneon, ymercury))

popt, pcov = np.polyfit(xdata, ydata, deg=4, cov=True)
popt = popt[::-1]
pcov = np.fliplr(np.transpose(np.fliplr(pcov)))

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

plt.plot(xneon, yneon, 'bo', label="Неоновая лампа", markersize=2)
plt.plot(xmercury, ymercury, 'rd', label="Ртутная лампа", markersize=4)
est_data = np.linspace(np.min(xdata), np.max(xdata), len(xdata))
plt.plot(est_data, f(est_data), 'g-', linewidth=1)
plt.title("Градуировка спектрометра")
plt.xlabel(r"$n$")
plt.ylabel(r"$\lambda, \si{\angstrom}$")
plt.legend()
plt.grid(visible=True)
plt.savefig("grad.pgf")

def table_data(x, y, out):
    tab = "\n".join((
    r"""\begin{tabular}{
        S[
            table-alignment-mode = none,
            round-mode = places,
            round-precision = 0,
        ]
        |
        S[
            table-alignment-mode = none,
            round-mode = places,
            round-precision = 0,
        ]
    }
    \toprule
    $n$ & $\lambda \text{, } \unit{\angstrom}$ \\
    \midrule""",

    "\n".join(f"{n} & {l} \\\\" for (n, l) in zip(x, y)),

    r"""\bottomrule
    \end{tabular}""",
    ))
    write_string(out, tab)

table_data(xneon, yneon, "neon.tab.tex")
table_data(xmercury, ymercury, "mercury.tab.tex")

balmer_names = (
    r"$H_\alpha$",
    r"$H_\beta$",
    r"$H_\gamma$",
    r"$H_\delta$",
)
balmer_m = np.arange(3, 3 + len(balmer_names))
balmer_n = np.loadtxt("balmer.txt")
balmer_n_err = np.full(balmer_n.size, n_err)
balmer_l = f(balmer_n)
balmer_err = errf(balmer_n, balmer_n_err)
balmer_real_l = (
    6563,
    4861,
    4340,
    4102,
)
balmer_R = 10**10 / (balmer_l * (1/4 - 1/(balmer_m**2)))
balmer_R_err = balmer_R * balmer_err / balmer_l
balmer_tab = "\n".join((
r"""\begin{tabular}{
    c
    |
    S[
        table-alignment-mode = none,
        round-mode = places,
        round-precision = 0,
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
        uncertainty-mode = separate,
        round-mode = uncertainty,
        round-precision = 1,
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
        uncertainty-mode = separate,
        round-mode = uncertainty,
        round-precision = 1,
        exponent-mode = scientific,
    ]
}
\toprule
& $m$ & $n$ & $\lambda\text{, }\unit{\angstrom}$ & $\lambda_{ref}\text{, }\unit{\angstrom}$ & $R\text{, }\unit{1/m}$\\
\midrule""",

"\n".join(f"{name} & {m} & {n} & {l} +- {err} & {lref} & {R} +- {R_err} \\\\" for (name, m, n, l, err, lref, R, R_err) in zip(
    balmer_names, balmer_m, balmer_n, balmer_l, balmer_err, balmer_real_l, balmer_R, balmer_R_err
)),

r"""\bottomrule
\end{tabular}""",
))
write_string("balmer.tab.tex", balmer_tab)

R_avg = np.average(balmer_R)
R_avg_eps = np.sqrt(np.sum(balmer_R_err ** 2)) / balmer_R_err.size

write_string("R.tex",
    "\n".join((
        r"""\begin{equation*}""",
        r"""\overline{R} = \qty[
            uncertainty-mode = separate,
            round-mode = uncertainty,
            round-precision = 1,
            exponent-mode = scientific,
        ]{""",
        f"{R_avg} +- {R_avg_eps}",
        r"""}{1/m}""",
        r"""\end{equation*}""",
    )))

from scipy.constants import speed_of_light, Planck, elementary_charge
err_k = 1

iodine_names = (
    r"$n_{1,0}$",
    r"$n_{1,5}$",
    r"$n_{\text{гр}}$",
)
iodine_n = np.array((2334, 2244, 1668,))
iodine_n_err = np.full(iodine_n.size, n_err)
iodine_l = f(iodine_n)
iodine_err = errf(iodine_n, iodine_n_err)
iodine_v = speed_of_light * 10**10 / iodine_l
iodine_v_err = iodine_v * iodine_err / iodine_l
iodine_hv = (Planck / elementary_charge) * iodine_v
iodine_hv_err = (Planck / elementary_charge) * iodine_v_err
iodine_tab = "\n".join((
r"""\begin{tabular}{
    c
    |
    S[
        table-alignment-mode = none,
        round-mode = places,
        round-precision = 0,
    ]
    |
    S[
        table-alignment-mode = none,
        uncertainty-mode = separate,
        round-mode = uncertainty,
        round-precision = 1,
    ]
    |
    S[
        table-alignment-mode = none,
        uncertainty-mode = separate,
        round-mode = uncertainty,
        round-precision = 1,
    ]
}
\toprule
& $n$ & $\lambda \text{, } \unit{\angstrom}$ & $h \nu \text{, } \unit{\electronvolt}$ \\
\midrule""",

"\n".join(f"{name} & {n} & {l} +- {err} & {hv} +- {hv_err} \\\\" for (name, n, l, err, hv, hv_err) in zip(
    iodine_names, iodine_n, iodine_l, iodine_err, iodine_hv, iodine_hv_err
)),

r"""\bottomrule
\end{tabular}""",
))
write_string("iodine.tab.tex", iodine_tab)

hv10 = iodine_hv[0]
hv15 = iodine_hv[1]
hvgr = iodine_hv[2]
Ea = 0.94
hv1 = 0.027
hv2 = (hv15 - hv10) / 5
hv10_err = iodine_hv_err[0]
hv15_err = iodine_hv_err[1]
hvgr_err = np.sqrt(iodine_hv_err[2]**2 + (err_k * hv2)**2)
hv2_err = np.sqrt(np.sum(np.array((hv15_err, hv10_err)) ** 2)) / 5
hvel = (3/2) * hv1 - (1/10) * hv15 + (11/10) * hv10
hvel_err = np.sqrt(np.sum(np.array(((1/10) * hv15_err, (11/10) * hv10_err, err_k * hv2)) ** 2))
D1 = hvgr - Ea
D1_err = hvgr_err
D2 = hvgr - hvel
D2_err = np.sqrt(np.sum(np.array((hvgr_err, hvel_err)) ** 2))
energies = (
    (r"$h \nu_{1,0}$", hv10, hv10_err),
    (r"$h \nu_{1,5}$", hv15, hv15_err),
    (r"$h \nu_{\text{гр}}$", hvgr, hvgr_err),
    (r"$E_a$", Ea, 0),
    (r"$h \nu_1$", hv1, 0),
    (r"$h \nu_2$", hv2, hv2_err),
    (r"$h \nu_{\text{эл}}$", hvel, hvel_err),
    (r"$D_1$", D1, D1_err),
    (r"$D_2$", D2, D2_err),
    (r"$h \nu_{\text{гр}_{ref}}$", 2.44, 0),
    (r"$D_{1_{ref}}$", 1.50, 0),
    (r"$D_{2_{ref}}$", 0.69, 0),
)
energies_tab = "\n".join((
r"""\begin{tabular}{
    c
    |
    S[
        table-alignment-mode = none,
        uncertainty-mode = separate,
        round-mode = uncertainty,
        round-precision = 1,
    ]
}
\toprule
& ${E} \text{, } \unit{\electronvolt}$ \\
\midrule""",

"\n".join(f"{name} & {hv} +- {hv_err} \\\\" for (name, hv, hv_err) in energies),

r"""\bottomrule
\end{tabular}""",
))
write_string("energies.tab.tex", energies_tab)
