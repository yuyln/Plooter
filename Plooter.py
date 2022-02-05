import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib import rcParams, cycler
from scipy.optimize import curve_fit


def FixPlot(lx, ly):
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 40
    rcParams['axes.linewidth'] = 1.1
    rcParams['axes.labelpad'] = 10.0
    plot_color_cycle = cycler('color', ['000000', 'FE0000', '0000FE', '008001', 'FD8000', '8c564b',
                                        'e377c2', '7f7f7f', 'bcbd22', '17becf'])
    rcParams['axes.prop_cycle'] = plot_color_cycle
    rcParams['axes.xmargin'] = 0
    rcParams['axes.ymargin'] = 0
    rcParams['legend.fancybox'] = False
    rcParams['legend.framealpha'] = 1.0
    rcParams['legend.edgecolor'] = "black"
    rcParams['legend.fontsize'] = 28
    rcParams['xtick.labelsize'] = 22
    rcParams['ytick.labelsize'] = 22

    rcParams['ytick.right'] = True
    rcParams['xtick.top'] = True

    rcParams['xtick.direction'] = "in"
    rcParams['ytick.direction'] = "in"

    rcParams.update({"figure.figsize": (lx, ly),
                    "figure.subplot.left": 0.177, "figure.subplot.right": 0.946,
                     "figure.subplot.bottom": 0.156, "figure.subplot.top": 0.965,
                     "axes.autolimit_mode": "round_numbers",
                     "xtick.major.size": 7,
                     "xtick.minor.size": 3.5,
                     "xtick.major.width": 1.1,
                     "xtick.minor.width": 1.1,
                     "xtick.major.pad": 5,
                     "xtick.minor.visible": True,
                     "ytick.major.size": 7,
                     "ytick.minor.size": 3.5,
                     "ytick.major.width": 1.1,
                     "ytick.minor.width": 1.1,
                     "ytick.major.pad": 5,
                     "ytick.minor.visible": True,
                     "lines.markersize": 10,
                     "lines.markeredgewidth": 0.8, 
                     "mathtext.fontset": "cm"})


def FixTicks(ax, minorx, minory, multx, multy):
    ax.yaxis.set_major_locator(MultipleLocator(multy))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minory))
    ax.xaxis.set_major_locator(MultipleLocator(multx))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minorx))


def FixScale(ax, datax, datay, pady=False, padx=False, mirrory=False, mirrorx=False, limx=None, limy=None):
    maxX = max(datax)
    minX = min(datax)
    maxY = max(datay)
    minY = min(datay)

    padMaxY = abs(maxY * pady)
    padMinY = abs(minY * pady)

    padMaxX = abs(maxX * padx)
    padMinX = abs(minX * padx)

    maxPadX = max(padMaxX, padMinX)
    maxPadY = max(padMaxY, padMinY)

    if mirrory:
        minY = minY - maxPadY
        maxY = maxY + maxPadY
    else:
        minY = minY - padMinY
        maxY = maxY + padMaxY

    if mirrorx:
        minX = minX - maxPadX
        maxX = maxX + maxPadX
    else:
        minX = minX - padMinX
        maxX = maxX + padMaxX

    if limx:
        maxX = limx[1]
        minX = limx[0]
    
    if limy:
        maxY = limy[1]
        minY = limy[0]

    ax.set_xlim([minX, maxX])
    ax.set_ylim([minY, maxY])
    return [minX, maxX], [minY, maxY]


def PlottaLine(ax, datax, datay, fmt="-", lw=2.5, funcx=lambda x: x, funcy=lambda y: y, label=None, 
               contx=0.5, conty=0.5, fit=False, funcfit=lambda x:x, labelfit=None, **kargs):
    x = []
    y = []
    tempx = [funcx(datax[0])]
    tempy = [funcy(datay[0])]
    maxfev = 1600
    method = 'lm'
    p0 = None
    colofA = 'colorfit' in kargs.keys()
    fitrangeX = [min(funcx(datax)), max(funcx(datax))]

    if "fitrangeX" in kargs.keys():
        fitrangeX = kargs["fitrangeX"]
        kargs.pop("fitrangeX")

    if 'maxfev' in kargs.keys():
        maxfev = kargs['maxfev']
        kargs.pop("maxfev")

    if 'method' in kargs.keys():
        method = kargs['method']
        kargs.pop("method")

    if 'p0' in kargs.keys():
        p0 = kargs['p0']
        kargs.pop("p0")

    for i in range(1, len(datax)):
        Desx = np.abs(funcx(datax[i]) - funcx(datax[i - 1])) > contx
        Desy = np.abs(funcy(datay[i]) - funcy(datay[i - 1])) > conty
        if Desx or Desy:
            x.append(tempx)
            y.append(tempy)
            tempx = []
            tempy = []
            continue
        tempx.append(funcx(datax[i]))
        tempy.append(funcy(datay[i]))
    x.append(tempx)
    y.append(tempy)
    i = 0
    if colofA:
        colorfit = kargs['colorfit']
        kargs.pop("colorfit")
    
    a = ax.plot(x[i], y[i], fmt, linewidth=lw, label=label, **kargs)
    ka = {k: kargs[k] for k in kargs if k != "color"}
    for i in range(1, len(x)):
        a = ax.plot(x[i], y[i], fmt, linewidth=lw, **ka, color=a[0].get_color())
    xret = [float(i.strip()) for i in str(x).replace("[", "").replace("]", "").split(",") if i.strip() != '']
    yret = [float(i.strip()) for i in str(y).replace("[", "").replace("]", "").split(",") if i.strip() != '']
    if fit:
        popt, pcov = curve_fit(funcfit, funcx(datax), funcy(datay), maxfev=maxfev, method=method, p0=p0)
        x_ = funcx(np.linspace(fitrangeX[0], fitrangeX[1], 1000))
        y_ = funcy(funcfit(x_, *popt))
        if not colofA:
            colorfit = a[0].get_color()
        dret = {"popt": popt, "pcov": pcov, "xb": xret, "yb": yret, "lineb": a}
        r = PlottaLine(ax, x_, y_, '--', color=colorfit, fit=False, label=labelfit, **ka)
        dret["xf"] = r["xb"]
        dret["yf"] = r["yb"]
        dret["linef"] = r["lineb"]
        return dret
    dret = {"xb": xret, "yb": yret, "lineb": a}
    return dret

def PlottaScatter(ax, datax, datay, funcx=lambda x: x, funcy=lambda y: y, label=None, size=20.0, **kargs):
    arg = [funcx(datax), funcy(datay)]
    a = ax.scatter(*arg, label=label, s=size, **kargs)
    return arg, a

    
def AddVerticalLine(ax, x, fmt='--', lw=2.5, **kargs):
    x_ = [x, x]
    y_ = ax.get_ylim()
    return ax.plot(x_, y_, fmt, linewidth=lw, **kargs)

def AddVHorizontalLine(ax, y, fmt='--', lw=2.5, **kargs):
    x_ = ax.get_xlim()
    y_ = [y, y]
    return ax.plot(x_, y_, fmt, linewidth=lw, **kargs)

def DecimalPlacesX(ax, n):
    f = f"%.{n}f"
    ax.xaxis.set_major_formatter(FormatStrFormatter(f))

def DecimalPlacesY(ax, n):
    f = f"%.{n}f"
    ax.yaxis.set_major_formatter(FormatStrFormatter(f))

def DecimalPlaces(ax, nx, ny):
    DecimalPlacesX(ax, nx)
    DecimalPlacesY(ax, ny)

def YLabel(ax, label, **kargs):
    ax.set_ylabel(label, **kargs)

def XLabel(ax, label, **kargs):
    ax.set_xlabel(label, **kargs)

def Labels(ax, labelx, labely):
    YLabel(ax, labely)
    XLabel(ax, labelx)

