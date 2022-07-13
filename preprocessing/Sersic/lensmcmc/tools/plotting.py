# Copyright (c) 2022, Conor O'Riordan

from matplotlib import rcParams, rcdefaults


def format_axes(a, log=False):
    if log:
        a.set_yscale('log')

    a.tick_params(axis='y', which='major', direction='in',
                  length=5.0, width=1.0, right=True)
    a.tick_params(axis='y', which='minor', direction='in',
                  length=3.0, width=1.0, right=True)
    a.tick_params(axis='x', which='major', direction='in',
                  length=5.0, width=1.0, top=True)
    a.tick_params(axis='x', which='minor', direction='in',
                  length=3.0, width=1.0, top=True)


def dark_mode():

    rcdefaults()
    rcParams['backend'] = 'Agg'
    rcParams['text.color'] = 'w'
    rcParams['axes.facecolor'] = [0.2]*3
    rcParams['figure.facecolor'] = [0.2]*3
    rcParams['savefig.facecolor'] = [0.2]*3
    rcParams['axes.edgecolor'] = [0.7]*3
    rcParams['ytick.direction'] = 'in'
    rcParams['ytick.right'] = True
    rcParams['ytick.color'] = [0.7]*3
    rcParams['xtick.direction'] = 'in'
    rcParams['xtick.top'] = True
    rcParams['xtick.color'] = [0.7]*3
    rcParams['axes.labelcolor'] = 'w'
