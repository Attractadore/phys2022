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

doc = setup()
doc.preamble.append(pl.Command('title', "Отчет по лаботраторной работе №2.1 \"Опыт Франка-Герца\""))
doc.preamble.append(pl.Package('hyperref'))
doc.preamble.append(pl.Package('float'))
doc.preamble.append(pl.Package('biblatex',options=['sorting=none']))
doc.preamble.append(pl.Command('addbibresource',arguments=["wiki.bib"]))

doc.append(pl.Command('maketitle'))

with doc.create(pl.Section("Цель работы")) as s:
    s.append("Найти энергию первого уровня атома гелия.")

with doc.create(pl.Section("Экспериментальная установка")) as s:
    s.append(NoEscape(r"""
        Трехэлектродную лампу заполняем разреженным гелием.
        При движении от катода к аноду, электрону могут либо упруго сталкиваться с ядром
        атома гелия почти без потерь энергии, либо неупруго соударяться с одним из его электронов,
        после чего налетающий электрон теряет часть своей энергии, а атом гелий возбуждается.

        Количество электронов, падающих на коллектор измеряется мультиметром.
        Между коллектором и анодом поддерживается небольшое задерживающее напряжение.
        Из-за этого электроны, потерявшие большую часть своей энергии на возбуждение атомов гелия,
        на коллектор не попадают.

        Первое замедление роста тока на коллекторе происходит, когда часть электронов
        неупруго сталкивается с атомами гелия 1 раз, второе -- когда 2 раза, и так далее.
        Таким образом, измеряя расстояние между локальными максимумами ВАХ коллектора, можно найти
        энергию возбуждения атома гелия.
    """))

with doc.create(pl.Section("Результаты измерений")) as s:
    s.append(NoEscape(r"""
        Первоначальные измерения проводятся грубо в динамическом режиме при помощи осциллографа,
        потом в статическом режиме цифровым мультиметром GwINSTEK GDM-8145.
    """))

    with doc.create(pl.Subsection("Измерения в динамическом режиме")) as ss:
        A = 2 * pq.electron_volt

        ss.append("Максимум и минимум фиксируются на экране осциллографа с ценой деления ")
        ss.append(pl.Math(data=["A=", pl.Quantity(2 * pq.volt, options={
            "round-mode": "figures",
            "round-precision": 1,
        })], inline=True, escape=False))
        ss.append(":")
        with doc.create(pl.Center()):
            pass

with doc.create(pl.Section("Заключение")) as s:
    pass

doc.append(pl.Command('printbibliography'))

doc.generate_pdf('report', clean_tex=False)
