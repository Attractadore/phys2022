#!/usr/bin/env python3
from pylatex.utils import NoEscape
import matplotlib.pyplot as plt
import numpy as np
import pylatex as pl
import quantities as pq

def setup():
    plt.rcParams.update({
        "text.latex.preamble": r"""
            \usepackage[main=russian, english]{babel}
            \usepackage{siunitx}
        """,
        "text.usetex": True,
        "font.family": "serif",
        "font.monospace": [],
        "font.sans-serif": [],
        "font.serif": [],
    })

    geometry_options = [
        "a4paper",
        "left=20mm",
        "right=20mm",
        "top=20mm",
    ]
    document_options = ['12pt']

    doc = pl.Document(geometry_options=geometry_options, document_options=document_options)
    doc.packages.append(pl.Package('babel', options=['english', 'russian']))
    doc.packages.append(pl.Package('parskip'))
    doc.preamble.append(pl.Command('date', ''))
    pl.config.active.booktabs = True

    pq.electron_volt = pq.UnitQuantity('electronvolt', pq.electron_volt.units, pq.electron_volt.symbol)

    return doc

number_options = {
    "uncertainty-mode": "separate",
    "round-mode": "uncertainty",
    "round-precision": 1,
    "separate-uncertainty-units": "single",
}

cite_key = "WebElements"
with open("wiki.bib", "w") as f:
    f.write(r"""
  @misc{ """ + cite_key + """,
    author = "Mark Winter",
    title = "WebElements: THE periodic table on the WWW",
    year = "2022",
    howpublished = "\\url{www.webelements.com/xenon/atom_sizes.html}",
  }
""")

doc = setup()
doc.preamble.append(pl.Command('title', "Отчет по лаботраторной работе №1.3 \"Эффект Рамзауэра\""))
doc.preamble.append(pl.Package('hyperref'))
doc.preamble.append(pl.Package('float'))
doc.preamble.append(pl.Package('biblatex',options=['sorting=none']))
doc.preamble.append(pl.Command('addbibresource',arguments=["wiki.bib"]))

doc.append(pl.Command('maketitle'))

with doc.create(pl.Section("Цель работы")) as s:
    doc.append("Найти размер электронной оболочки атома ксенона.")

atom_radius_tex = r"\frac{h \sqrt{5}}{\sqrt{32 m (E_2 - E_1)}}"
def atom_radius_f(h=None, m=None, E_1=None, E_2=None):
    return (h * (5 ** .5)) / ((32 * m * (E_2 - E_1)) ** .5)

potential_gap_tex = r"\frac{4}{5} E_2 - \frac{9}{5} E_1"
def potential_gap_f(E_1=None, E_2=None):
    return 4/5 * E_2 - 9/5 * E_1

D_max_tex = r"\sqrt{ \frac{2m (E_n + U_0)}{\hbar^2} } l = \pi n"
l_from_E_1_tex = r"\frac{1}{2} \frac{h}{\sqrt{2m \left( E_1 + U_0 \right)}}"
def l_from_E_1_f(h=None, m=None, E_1=None, U_0=None):
    return .5 * h / (2 * m * (E_1 + U_0)) ** .5

with doc.create(pl.Section("Теоретические сведения")) as s:
    doc.append(NoEscape(r"""
        Эффектом Рамзауера называется расхождение результатов наблюдений
        рассеяния электронов на атомах инертного газа с предсказаниями
        классической электромагнитной теории.

        Рассмотрим рассеяние электронов на атоме тяжелого инертного газа с точки
        зрения квантовой механики.
        Для качественного анализа можно считать,
        что при прохождении мимо атома, электрон попадает в
        потенциальную яму глубиной $U_0$, сопоставимой по длине $l$ с размером
        атома.
        Решив уравнение Шредингера, можно найти условие для максимума
        и минимума коэффициента прохождения электрона с энергией $E$ через яму,
        что соответствует максимуму и минимуму вероятности рассеяния электрона $w$:
    """))
    doc.append(pl.Math(data=[r"k l = ", D_max_tex], escape=False))
    doc.append(pl.Math(data=[
        r"l = \frac{1}{2} \frac{h}{\sqrt{2m \left( E_1 + U_0 \right)}}"], escape=False
    ))
    doc.append(pl.Math(data=[f'l={l_from_E_1_tex}'], escape=False))
    doc.append(pl.Math(data=[r"k l = ", D_max_tex], escape=False))
    doc.append(NoEscape(r"""
        Измерив энергию первого максимума прохождения $E_1$ и первого минимума
        прохождения $E_2$, можно найти $l$ и $U_0$:
    """))
    doc.append(pl.Math(data=[f'l={atom_radius_tex}'], escape=False))
    doc.append(pl.Math(data=[f'U_0={potential_gap_tex}'], escape=False))

with doc.create(pl.Section("Экспериментальная установка")) as s:
    with doc.create(pl.Figure(position=NoEscape('H'))) as fig:
        fig.add_image('sys.png')
        fig.add_caption('Тиратрон ТГ3-01/1.3Б')
    doc.append(NoEscape(r"""
        Для изучения эффекта Рамзауэра используем Тиратрон ТГ3-01/1.3Б,
        заполненный инертным газом.
        Электроны излучаются с катода $5$ и ускоряются напряжением $V$,
        приложенным между ним и ближайшей сеткой $1$. Потенциалы сеток $1$, $2$
        и анода $6$ примерно равны, поэтому поле между ними мало. Рассеявшиеся
        электроны попадают на сетки, а прошедший -- попадают на анод и образуют ток $I$.

        Посмотрим, как происходит рассеивание электронов при движении от катода к аноду.
        Пусть $n$ -- плотность газа в тиратроне, $\Delta_a$ -- поперечное сечение атома этого газа,
        $S$ -- площадь сечения тиратрона.
        Тогда в тонком слое толщиной $dx$ содержится $d\nu$ = $n S dx$ атомов газа эффективным сечением
        $\Delta = d \nu \Delta_a$. Вероятность того, что электрон наткнется на атом, равна $\frac{\Delta}{S}$.
        Тогда вероятность того, что электрон рассеется при прохождении тонкого слоя равна $W(V) = \frac{\Delta}{S} w(V)$.
        При прохождении тонкого слоя газа $N$ электронами, рассеются $dN$ электронов:
    """))
    doc.append(pl.Math(data=[r'dN = - W(V) N = - n \Delta_a w(V) N dx'], escape=False))
    doc.append(NoEscape(r"""
        Интегрируя от $0$ до длины тиратрона $L$ и заменяя число электронов $N$ на ток $I$, получаем:
    """))
    doc.append(pl.Math(data=[r'I(V) = I_0 e^{-C w \left( V \right)}, C = L n \Delta_a'], escape=False))
    doc.append(NoEscape(r"""
        Здесь $I(V)$ - ток на аноде, $I_0$ - ток на катоде.

        Измерив ВАХ тиратрона, можно найти вероятность рассеяния электронов:
    """))
    doc.append(pl.Math(data=[r'w = - \frac{1}{C}\ln{\frac{I(V)}{I_0}}'], escape=False))
    doc.append(NoEscape(r"""
        Максимальное значение коэффициента прохождения электрона равно $1$, следовательно $w(V_1) = 0$, а $I_0 = I(V_1)$.
    """))

with doc.create(pl.Section("Результаты измерений")) as s:
    doc.append(NoEscape(r"""
        Напряжение накала $U_{\text{н}} = \qty[]{2.66}{\volt}$.
        Первоначальные измерения проводятся грубо в динамическом режиме и
        при помощи осциллографа, потом в статическом режиме цифровым мультиметром GwINSTEK GDM-8145.
    """))

    with doc.create(pl.Subsection("Измерения в динамическом режиме")) as ss:
        A = 2 * pq.electron_volt
        V_f_1 = 1.5
        V_f_2 = 3.5
        V_b_1 = 1.3
        V_b_2 = 4.2
        E_1_q = pq.UncertainQuantity(1.4, '', .2) * A
        E_2_q = pq.UncertainQuantity((3.5 + 4.2) / 2, '', .7) * A

        ss.append("Максимум и минимум фиксируются на экране осциллографа с ценой деления ")
        ss.append(pl.Math(data=["A=", pl.Quantity(2 * pq.volt, options={
            "round-mode": "figures",
            "round-precision": 1,
        })], inline=True, escape=False))
        ss.append(":")
        with ss.create(pl.Center()):
            S = r"""S[
                table-alignment-mode = none,
                round-mode = figures,
                round-precision = 2,
            ]"""
            with ss.create(pl.Tabular(f'c|{S}|{S}', width=3)) as table:
                table.add_row((NoEscape(""), NoEscape(r'$V_{1} \text{, } \unit{\text{дел}}$'), NoEscape(r'$V_{2} \text{, } \unit{\text{дел}}$')))
                table.add_hline()
                table.add_row(r"Ход луча вперед", V_f_1, V_f_2)
                table.add_row(r"Ход луча назад", V_b_1, V_b_2)

        E_1_q = pq.UncertainQuantity((V_f_1 + V_b_1) / 2, '', abs(V_f_1 - V_b_1)) * A
        E_2_q = pq.UncertainQuantity((V_f_2 + V_b_2) / 2, '', abs(V_f_2 - V_b_2)) * A
        U_0_q = 2.5 * pq.electron_volt
        ss.append(NoEscape(r"""
            Положение каждого экстремума оценим как средние от измеренного во время хода луча осциллографа
            вперед и назад, а погрешность как разность:
        """))
        ss.append(pl.Math(data=['E_1=', pl.Quantity(E_1_q, options=number_options)], escape=False))
        ss.append(pl.Math(data=['E_2=', pl.Quantity(E_2_q, options=number_options)], escape=False))

        ss.append(NoEscape(r"Вычислим $l$ при известном $U_0$:"))
        ss.append(pl.Math(data=['U_0=', pl.Quantity(U_0_q, options={
            "round-mode": "places",
            "round-precision": 1,
        })], escape=False))
        l_q = pq.UncertainQuantity(
            l_from_E_1_f(h=pq.constants.h, m=pq.constants.electron_mass, E_1=E_1_q, U_0=pq.UncertainQuantity(U_0_q)),
            pq.angstrom
        )
        l = pl.Quantity(l_q, options=number_options)
        ss.append(pl.Math(data=[f'l={l_from_E_1_tex}=', l], escape=False))
        l_dyn_know_U_0 = l

        ss.append(NoEscape(r"И через $E_1$ и $E_2$:"))
        l_q = pq.UncertainQuantity(
            atom_radius_f(h=pq.constants.h, m=pq.constants.electron_mass, E_1=E_1_q, E_2=E_2_q),
            pq.angstrom
        )
        l = pl.Quantity(l_q, options=number_options)
        ss.append(pl.Math(data=[f'l={atom_radius_tex}=', l], escape=False))
        l_dyn = l

        """
        ss.append(NoEscape(r"Глубина потенциальной ямы:"))
        U_0_q = potential_gap_f(E_1 = E_1_q, E_2 = E_2_q)
        U_0 = pl.Quantity(U_0_q, options=number_options)
        ss.append(pl.Math(data=[f"U_0={potential_gap_tex}=", U_0], escape=False))
        """

    with doc.create(pl.Subsection("Измерения в статическом режиме")) as ss:
        err_V_q = pq.Quantity(0.01, pq.volt)
        err_I_q = pq.Quantity(0.1, pq.millivolt)
        ss.append(NoEscape(r"""
            Производитель заявляет, что точность мультиметра GDM-8145 --
            $\pm \left( 0.03\% + 4 \text{ цифры дисплея} \right)$.
            Измерение $V$ происходит в диапазоне $0-20 \unit{\volt}$,
            значит систематическую погрешность можно оценить как
        """))
        ss.append(pl.Math(data=["\sigma_V=", pl.Quantity(err_V_q)], escape=False, inline=True))
        ss.append(NoEscape(r""".
            Ток $I$ измеряется как напряжение на резисторе
            $R = \qty[]{100}{\kilo\ohm}$ в диапазоне $0-200 \unit{\milli\volt}$,
            поэтому
        """))
        ss.append(pl.Math(data=["\sigma_I=", pl.Quantity(err_I_q)], escape=False, inline=True))
        ss.append(".")

        data = np.array([
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
        ])
        data = data.reshape(data.size // 2, 2)

        with ss.create(pl.Center()):
            S = r"""S[
                table-alignment-mode = none,
                round-mode = figures,
                round-precision = 3,
            ]"""
            with ss.create(pl.Tabular(f'{S}|{S}', width=2)) as table:
                table.add_row((NoEscape(r'$V \text{, } \unit{\volt}$'), NoEscape(r'$I \text{, } \unit{\milli\volt}$')))
                table.add_hline()
                for r in data:
                    table.add_row(r)

        data_V, data_I = data.transpose()

        """
        def par_ext_err(p, V):
            a, b, _ = p
            x_ext = -b/2/a
            v = np.abs(np.array([1/a, 1/b, 0]))
            x_err = np.sqrt(np.dot(v.transpose(), np.dot(V, v))) * np.abs(x_ext)
            return x_ext, x_err

        i = np.where(data_I == 22.4)[0][0]
        w = 2
        lo, hi = (i - w, i + w + 1)
        p, V = np.polyfit(data_V[lo: hi], data_I[lo: hi], 2, cov=True)
        Vmax, Vmax_err = par_ext_err(p, V)
        I_0_q = p[2] - p[1] ** 2 / 4 / p[0]
        data_Cw = -np.log(data_I / I_0_q)
        max_est_x = np.linspace(data_V[lo], data_V[hi - 1], 2 * (hi - lo))
        max_est_y = np.dot(np.column_stack((max_est_x * max_est_x, max_est_x, np.ones(max_est_x.shape))), p)

        i = np.where(data_I == 9.67)[0][0]
        w = 5
        lo, hi = (i - w, i + w + 1)
        p, V = np.polyfit(data_V[lo: hi], data_I[lo: hi], 2, cov=True)
        Vmin, Vmin_err = par_ext_err(p, V)
        min_est_x = np.linspace(data_V[lo], data_V[hi - 1], 2 * (hi - lo))
        min_est_y = np.dot(np.column_stack((min_est_x * min_est_x, min_est_x, np.ones(min_est_x.shape))), p)
        """

        I_0_q = 22.4
        data_Cw = -np.log(data_I / I_0_q)
        i = np.where(data_I == I_0_q)[0][0]
        Vmax = data_V[i]
        l = r = i
        while abs(data_I[l] - data_I[i]) <= err_I_q:
            l -= 1
        while abs(data_I[r] - data_I[i]) <= err_I_q:
            r += 1
        Vmax_err = max(Vmax - data_V[l], data_V[r] - Vmax)

        i = np.where(data_I == 9.67)[0][0]
        Vmin = data_V[i]
        l = r = i
        while abs(data_I[l] - data_I[i]) <= err_I_q:
            l -= 1
        while abs(data_I[r] - data_I[i]) <= err_I_q:
            r += 1
        Vmin_err = max(Vmin - data_V[l], data_V[r] - Vmin)

        with ss.create(pl.Figure(position=NoEscape('H'))) as plot:
            plt.figure(clear=True)
            plt.plot(data_V, data_I, 'o')
            plt.xlabel(r"$V \text{, } \unit{V}$")
            plt.ylabel(r"$I \text{, } \unit{mV}$")
            plt.grid()
            plot.add_plot()
            plot.add_caption(r"ВАХ тиратрона")

        E_1_q = pq.UncertainQuantity(Vmax, pq.electron_volt, Vmax_err)
        E_2_q = pq.UncertainQuantity(Vmin, pq.electron_volt, Vmin_err)
        E_1 = pl.Quantity(E_1_q, options=number_options)
        E_2 = pl.Quantity(E_2_q, options=number_options)
        I_0 = pl.Quantity(I_0_q * pq.millivolt, options={
            "round-mode": "figures",
            "round-precision": 3,
        })

        ss.append(NoEscape("""
        Найдем максимум и минимум.
        Погрешность оценим как максимальное расстояние между экстремумом и точкой, чье значение
        не отличается с учетом погрешности:
        """))
        ss.append(pl.Math(data=["E_1=", E_1], escape=False))
        ss.append(pl.Math(data=["E_2=", E_2], escape=False))
        ss.append(NoEscape(r"""
        Примерное значение тока на аноде:
        """))
        ss.append(pl.Math(data=[r"I_0 \approx", I_0], escape=False))

        r"""
        ss.append(NoEscape("Для максимума и минимума учтем систематическую погрешность:"))
        E_1_q += pq.UncertainQuantity(0, pq.e * err_V_q.units, float(err_V_q))
        E_2_q += pq.UncertainQuantity(0, pq.e * err_V_q.units, float(err_V_q))
        E_1 = pl.Quantity(E_1_q, options=number_options)
        E_2 = pl.Quantity(E_2_q, options=number_options)
        ss.append(pl.Math(data=["E_1=", E_1], escape=False))
        ss.append(pl.Math(data=["E_2=", E_2], escape=False))
        """

        ss.append(NoEscape("Теперь рассчитаем размер электронной оболочки атома $l$ и глубину потенциальной ямы $U_0$:"))
        l_q = atom_radius_f(h = pq.constants.h, m = pq.constants.electron_mass, E_1=E_1_q, E_2=E_2_q)
        l_q = pq.UncertainQuantity(l_q, pq.angstrom)
        l = pl.Quantity(l_q, options=number_options)
        U_0_q = potential_gap_f(E_1 = E_1_q, E_2 = E_2_q)
        U_0 = pl.Quantity(U_0_q, options=number_options)
        l_stat = l

        ss.append(pl.Math(data=[f"l={atom_radius_tex}=", l], escape=False))
        ss.append(pl.Math(data=[f"U_0={potential_gap_tex}=", U_0], escape=False))

        ss.append("Оценим, где будут находится следующие 2 максимума:")
        ss.append(pl.Math(data=[D_max_tex], escape=False))
        ss.append(pl.Math(data=[r"E_n = \frac{n^2 h^2}{8m l^2} - U_0"], escape=False))

        def E_n_f(n=0, m=None, l=None, h=None, U_0=None):
            return (h ** 2 * n ** 2) / (8 * m * l ** 2) - U_0

        E_n_2_q = E_n_f(n=2, m=pq.constants.electron_mass, l=l_q, h=pq.constants.h, U_0=U_0_q)
        E_n_3_q = E_n_f(n=3, m=pq.constants.electron_mass, l=l_q, h=pq.constants.h, U_0=U_0_q)
        E_n_2_q = pq.UncertainQuantity(E_n_2_q, pq.electron_volt)
        E_n_3_q = pq.UncertainQuantity(E_n_3_q, pq.electron_volt)
        E_n_2 = pl.Quantity(E_n_2_q, options=number_options)
        E_n_3 = pl.Quantity(E_n_3_q, options=number_options)

        ss.append(pl.Math(data=['E_{n|n=2}=', E_n_2], escape=False))
        ss.append(pl.Math(data=['E_{n|n=3}=', E_n_3], escape=False))

        ss.append("Построим график вероятности рассеяния электрона:")
        with ss.create(pl.Figure(position='H')) as plot:
            plt.figure(clear=True)
            idx = np.where(data_I > 0)
            plt.plot(data_V[idx], data_Cw[idx], 'o')
            plt.xlabel(r"$V \text{, } \unit{V}$")
            plt.ylabel(r"$Cw$")
            plt.grid()
            plot.add_plot()
            plot.add_caption(r"Вероятность рассеяния электрона")

    with doc.create(pl.Subsection("Анализ результатов")) as ss:
        with ss.create(pl.Center()):
            with ss.create(pl.Tabular('c|c', width=2)) as table:
                table.add_row(("", NoEscape(r'$l \unit{}$')))
                table.add_hline()
                r_tab_q = pq.Quantity(1.30, pq.angstrom)
                l_tab_q = 2 * r_tab_q
                l_tab = pl.Quantity(l_tab_q, options=number_options)
                table.add_row((NoEscape(r"Удвоенный ковалентный радиус \textsuperscript{\cite{" + cite_key + r"}}"), l_tab))
                table.add_row((NoEscape(r"Динамический режим при известном $U_0$"), l_dyn_know_U_0))
                table.add_row((NoEscape(r"Динамический режим"), l_dyn))
                table.add_row((NoEscape(r"Статический режим"), l_stat))

        ss.append(NoEscape(r"""
            Полученные результаты сопоставимы с табличным значением удвоенного ковалентного радиуса атома ксенона.
            Рассчитанный при заданном значении $U_0$ размер атома совпадает с табличным значением.
            Измерения в статическом режиме точнее измерений в динамическом режиме, поэтому далее будем считать,
            что
        """))
        ss.append(pl.Math(data=[f"l=", l], inline=True, escape=False))
        ss.append(", однако это значение не совпадает с табличным, вероятно из-за выбранной модели.")

with doc.create(pl.Section("Заключение")) as s:
    s.append(NoEscape(r"""
        В ходе работы удалось оценить размер электронной оболочки атома ксенона
    """))
    s.append(pl.Math(data=["l=", l_stat], escape=False, inline=True))
    s.append(".")

doc.append(pl.Command('printbibliography'))

doc.generate_pdf('report', clean_tex=False)
