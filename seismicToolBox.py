"""
A python module for plotting and manipulating seismic data and headers,
kirchhoff migration and more. Contains the following functions

load_header             : load mat file header
load_segy               : load segy dataset
sorthdr                 : sort seismic header
sortdata                : sort seismic data
selectCMP               : select a CMP gather with respect to its midpoint position
analysefold             : Positions of gathers and their folds
imageseis               : Interactive seismic image
wiggle                  : Seismic wiggle plot
plothdr                 : Plots header data
semblanceWiggle         : Interactive semblance plot for velocity analysis
apply_nmo               : Applies nmo correction
nmo_v                   : Applies nmo to single CMP gather for constant velocity
nmo_vlog                : Applied nmo to single CMP gather for a 1D time-velocity log
nmo_stack               : Generates a stacked zero-offset section
stackplot               : Stacks all traces in a gather and plots the stacked trace
generatevmodel2         : Generates a 2D velocity model
kirk_mig                : Kirkhoff migration
time2depth_trace        : time-to-depth conversion for a single trace in time domain
time2depth_section      : time-to-depth conversion for a seismic section in time domain
agc                     : applies automatic gain control for a given dataset.

(C) Nicolas Vinard and Musab al Hasani, 2020, v.0.0.7

- 9.9.2020 added load_header and load_segy
- 31.01.2020 added nth-percentile clipping
- 16.01.2020 Added clipping in wiggle function
- added agc functions
- Fixed semblanceWiggle sorting error


Many of the functions here were developed by CREWES and were originally written in MATLAB.
In the comments of every function we translated from CREWES we refer to its function name
and the CREWES Matlab library at www.crewes.org.
Other functions were translated from MATLAB to Python and originally written by Max Holicki
for the course "Geophyiscal methods for subsurface characterization" tought at TU Delft.

"""

import struct, sys
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
from typing import Tuple
#from tqdm import tnrange
#from utils import segypy
from scipy.io import loadmat


### LOAD HEADER OR SEGY DATA ####

def load_header(file_path)->np.ndarray:
    """

    header = load_header(path_to_file)

    Parameters
    ----------
    file_path: string
        Path to mat file

    Returns
    -------
    SH: np.ndarray of shape (5, # traces)
        Header containing information of shot, receiver, CMP and offset positions

    """
    SH = loadmat(file_path)

    return SH['H_SHT']

def load_segy(file_path)->tuple([np.ndarray, np.ndarray, np.ndarray]):
    """

    Data, SH, STH  = load_segy(path_to_file)

    Parameters
    ----------
    file_path: string
        Path to mat file

    Returns
    -------
    Data: np.ndarray of shape (# time samples, # traces)
        Seismic data
    SH: np.ndarray of shape (5, # traces)
        Header containing information of shot, receiver, CMP and offset positions
    STH: np.ndarray
        Trace header

    """

    segyDataset = segypy.readSegy(file_path)
    Data = segyDataset[0]
    SH = segyDataset[2]
    STH = segyDataset[1]

    return Data, SH, STH


##################################################
########## HEADER AND DATA MANIPULATION ##########
##################################################

def sorthdr(H_SHT: np.ndarray, sortkey1: int, sortkey2 = None)->np.ndarray:
    """
    sorted_header = sorthdr(H_SHT, sortkey1, sortkey2 = None)

    Sorts the input header according to the sortkey1 value as primary sort,
    and within sortykey1 the header is sorted according to sortkey2.

    Valid values for sortkey are:
        1 = Common Shot
        2 = Common Receiver
        3 = Common Midpoint (CMP)
        4 = Common Offset

    Parameters
    ----------
    H_SHT: np.ndarray of shape (5, # traces)
        Header containing information of shot, receiver, CMP and offset positions
    sortkey1: int
        Primary sort key by which to sort the header

    Optional parameters
    -------------------
    sortkey2: int
        Secondary sort key by which to sort the header

    Returns
    -------
    H_SRT: np.ndarray of shape (5, # traces)
        Sorted header

    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    """

    if sortkey2 is None:
        if sortkey1 == 4:
            sortkey2 = 3
        if sortkey1 == 1:
            sortkey2 = 2
        if sortkey1 == 2:
            sortkey2 = 1
        if sortkey1 == 3:
            sortkey2 = 4


    index_sort = np.lexsort((H_SHT[sortkey2, :], H_SHT[sortkey1, :]))

    H_SRT = H_SHT[:, index_sort]

    return H_SRT


def sortdata(
    data: np.ndarray,
    H_SHT: np.ndarray,
    sortkey1: int,
    sortkey2 = None
    )->tuple([np.ndarray, np.ndarray]):

    """
    sorted_data, sorted_header = sortdata(data, H_SHT, sortkey1, sortkey2 = None):

    Sorts data using the data's header. Sorting order is defined according to the
    sortkey1 value as primary sort, and within sortykey1 it is is sorted
    again according to sortkey2.

    Valid values for sortkey are:

        1 = Common Shot
        2 = Common Receiver
        3 = Common Midpoint (CMP)
        4 = Common Offset

    Parameters
    ----------
    data: np.ndarray of shape (# time samples, # traces)
        The seismic data
    H_SHT: np.ndarray of shape (5, # traces)
        Header containing information of shot, receiver, CMP and offset positions

    Optional parameters
    -------------------
    sortkey2: int
        Secondary sort key by which to sort the header

    Returns
    -------
    sorted_data: np.ndarray of shape (# time samples, # traces)
        Sorted seismic data
    sorted_header: np.ndarray of shape (5, # traces)
        Sorted header


    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    """

    if sortkey2 is None:
        if sortkey1 == 4:
            sortkey2 = 3
        if sortkey1 == 1:
            sortkey2 = 2
        if sortkey1 == 2:
            sortkey2 = 1
        if sortkey1 == 3:
            sortkey2 = 4

    ns, n_traces = data.shape

    # Put header on top of data
    headered_data = np.append(H_SHT, data, axis=0)
    index_sort = np.lexsort((headered_data[sortkey2, :], headered_data[sortkey1, :]))
    sorted_headerData = headered_data[:, index_sort]
    sorted_data = sorted_headerData[len(H_SHT):, :]
    sorted_header = sorted_headerData[:len(H_SHT), :]

    return sorted_data, sorted_header


def selectCMP(
    CMPsorted: np.ndarray,
    H_CMP: np.ndarray,
    midpnt: float
    )->tuple([np.ndarray, np.ndarray]):

    """

    CMPGather, H_CMPGather = selectCMP(CMPsorted, H_CMP, midpnt):

    This function selects a CMP gather according to its midpoint position.
    Midpoints can be found with the function analysefold.

    Parameters
    ----------
    CMPsorted: np.ndarray of shape (# time samples, # traces)
        CMP sorted seismic data
    H_CMP: np.ndarray of shape (5, # traces)
        CMP sorted header
    midpnt: float
        Midpoint of the CMP-gather you want to plot

    Returns
    -------
    CMPGather: np.ndarray
        Selected CMP-gather
    H_CMPgather: np.ndarray
        Header of selectred CMP gather

    see also sortdata, anaylsefold


    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    """

    # Read the amount of time-samples and traces from the shape of the datamatrix
    nt, ntr = CMPsorted.shape

    # initialise arrays
    CMPgather = np.empty(CMPsorted.shape)
    H_CMPgather = np.empty(H_CMP.shape)

    # CMP-gather trace counter initialisation:
    # l is the number of traces in a CMP gather
    l = 0

    # Scan the CMP-sorted dataset for traces with the correct midpoint and put those traces in a cmpgather.

    for i in range(0,ntr):

        if H_CMP[3,i] == midpnt:

            CMPgather[:,l] = CMPsorted[:,i]
            H_CMPgather[:,l] = H_CMP[:,i]
            l = l + 1

    return CMPgather[:,:l], H_CMPgather[:,:l]



##################################################
########## HEADER AND DATA VISUALIZATION #########
##################################################

def analysefold(H_SHT: np.ndarray, sortkey: int
    )->tuple([np.ndarray, np.ndarray]):

    """
    positions, folds = analysefold(H_SHT, sortkey)

    This function gives the positions of the gathers, such as CMP's or
    Common-Offset gathers, as well as their folds. Furthermore, a
    crossplot is generated.

    Analysefold analyzes the fold of a dataset according to the value of sortkey:
        1 = Common Shot
        2 = Common Receiver
        3 = Common Midpoint (CMP)
        4 = Common Offset

    Parameters
    ----------
    H_SHT: np.ndarray of shape (5, # traces)
        Shot-sorted data header
    sortkey: int
        Sorting key

    Returns
    -------
    positions: np.ndarray
        Gather positions
    folds: np.ndarray
        Gather-folds


    Translated to Python from Matlab by Musab al Hasani and Nicolas Vinard, 2019

    """

    # Read the amount of time-samples and traces from the shape of the datamatrix
    nt, ntr = H_SHT.shape

    # Sort the header
    H_SRT = sorthdr(H_SHT, sortkey)

    # Midpoint initialization, midpoint distance and fold
    gather_positions = H_SRT[1,0]
    gather_positions = np.array(gather_positions)
    gather_positions = np.reshape(gather_positions, (1,1))
    gather_folds = 1
    gather_folds = np.array(gather_folds)
    gather_folds = np.reshape(gather_folds, (1,1))

    # Gather trace counter initialization
    # l is the amount of traces in a gather
    l = 0

    # Distance counter initialization
    # m is the amount of distances in the sorted dataset
    m = 0

    for k in range(1, ntr):

        if H_SRT[sortkey, k] == gather_positions[m]:
            l = l + 1
        else:
            if m == 0:
                gather_folds[0,0] = l
            else:
                gather_folds = np.append(gather_folds, l)

            m = m + 1
            gather_positions = np.append(gather_positions, H_SRT[sortkey, k])
            l = 0

    gather_folds = gather_folds+1

    # Remove first superfluous entry in gather_positions
    gather_positions = gather_positions[1:]

    # Make a plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(gather_positions, gather_folds, 'x')
    ax.set_title('Amount of traces per gather')
    ax.set_xlabel('gather-distance [m]')
    ax.set_ylabel('fold')
    fig.tight_layout()

    return gather_positions, gather_folds

def imageseis(DataO: np.ndarray, x=None, t=None, gain=1, perc=100):

    """
    imageseis(Data, x=None, t=None, maxval=-1, gain=1, perc=100):

    This function generates a seismic image plot including interactive
    handles to apply a gain and a clip

    Parameters
    ----------
    Data: np.ndarray of shape (# time samples, # traces)
        Seismic data

    Optional parameters
    -------------------
    gain: float
        Apply simple gain
    x: np.ndarray of shape Data.shape[1]
        x-coordinates to Plot
    t: np.ndarray of shape Data.shap[0]
        t-axis to plot
    perc: float
        nth parcintile to be clipped

    Returns
    -------
    Seismic image

    Adapted from segypy (Thomas Mejer Hansen, https://github.com/cultpenguin/segypy/blob/master/segypy/segypy.py)

    Musab and Nicolas added interactive gain and clip, 2019


    """

    # Make a copy of the original, so that it won't change the original one ouside the scope of the function
    Data = copy.copy(DataO)

    # calculate value of nth-percentile, when perc = 100, data won't be clipped.
    nth_percentile = np.abs(np.percentile(Data, perc))

    # clip data to the value of nth-percintile
    Data = np.clip(Data, a_min=-nth_percentile, a_max = nth_percentile)

    ns, ntraces = Data.shape
    maxval = -1
    Dmax = np.max(Data)
    maxval = -1*maxval*Dmax

    if t is None:
        t = np.arange(0, ns)
        tLabel = 'Sample number'
    else:
        t = t
        tc = t
        tLabel = 'Time [s]'
        if len(t)!=ns:
            print('Error: time array not of same length as number of time samples in data \n Samples in data: {}, sample in input time array: {}'.format(ns, len(t)))
            sys.exit()

    if x is None:
        x = np.arange(0, ntraces) +1
        xLabel = 'Trace number'
    else:
        x = x
        xLabel = 'Distance [m]'
        if len(x)!=ntraces:
            print('Error: x array not of same length as number of trace samples in data \n Samples in data: {}, sample in input x array: {}'.format(ns, len(t)))

    plt.subplots_adjust(left=0.25, bottom=0.3)
    img = plt.pcolormesh(x, t, Data*gain, vmin=-1*maxval, vmax=maxval, cmap='seismic')
    cb = plt.colorbar()
    plt.axis('auto')
    plt.xlabel(xLabel)
    plt.ylabel(tLabel)
    plt.gca().invert_yaxis()

    # Add interactice widgets
    # Defines position of the toolbars
    ax_cmax  = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_cmin = plt.axes([0.25, 0.1, 0.5, 0.03])
    ax_gain  = plt.axes([0.25, 0.05, 0.5, 0.03])

    s_cmax = Slider(ax_cmax, 'max clip ', 0, np.max(np.abs(Data)), valinit=np.max(np.abs(Data)))
    s_cmin = Slider(ax_cmin, 'min clip', -np.max(np.abs(Data)), 0, valinit=-np.max(np.abs(Data)))
    s_gain = Slider(ax_gain, 'gain', gain, 10*gain, valinit=gain)

    def update(val, s=None):
        _cmin = s_cmin.val/s_gain.val
        _cmax = s_cmax.val/s_gain.val
        img.set_clim([_cmin, _cmax])
        plt.draw()

    s_cmin.on_changed(update)
    s_cmax.on_changed(update)
    s_gain.on_changed(update)

    return img

def wiggle(
    DataO: np.ndarray,
    x=None,
    t=None,
    skipt=1,
    lwidth=.5,
    gain=1,
    typeD='VA',
    color='red',
    perc=100):

    """
    wiggle(DataO, x=None, t=None, maxval=-1, skipt=1, lwidth=.5, gain=1, typeD='VA', color='red', perc=100)

    This function generates a wiggle plot of the seismic data.

    Parameters
    ----------
    DataO: np.ndarray of shape (# time samples, # traces)
        Seismic data

    Optional parameters
    -------------------
    x: np.ndarray of shape Data.shape[1]
        x-coordinates to Plot
    t: np.ndarray of shape Data.shap[0]
        t-axis to plot
    skipt: int
        Skip trace, skips every n-th trace
    ldwidth: float
        line width of the traces in the figure, increase or decreases the traces width
    typeD: string
        With or without filling positive amplitudes. Use type=None for no filling
    color: string
        Color of the traces
    perc: float
        nth parcintile to be clipped

    Returns
    -------
    Seismic wiggle plot

    Adapted from segypy (Thomas Mejer Hansen, https://github.com/cultpenguin/segypy/blob/master/segypy/segypy.py)


    """
    # Make a copy of the original, so that it won't change the original one ouside the scope of the function
    Data = copy.copy(DataO)

    # calculate value of nth-percentile, when perc = 100, data won't be clipped.
    nth_percentile = np.abs(np.percentile(Data, perc))

    # clip data to the value of nth-percentile
    Data = np.clip(Data, a_min=-nth_percentile, a_max = nth_percentile)

    ns = Data.shape[0]
    ntraces = Data.shape[1]

    fig = plt.gca()
    ax = plt.gca()
    ntmax=1e+9 # used to be optinal

    if ntmax<ntraces:
        skipt=int(np.floor(ntraces/ntmax))
        if skipt<1:
                skipt=1

    if x is not None:
        x=x
        ax.set_xlabel('Distance [m]')
    else:
        x=range(0, ntraces)
        ax.set_xlabel('Trace number')

    if t is not None:
        t=t
        yl='Time [s]'
    else:
        t=np.arange(0, ns)
        yl='Sample number'

    dx = x[1]-x[0]

    Dmax = np.nanmax(Data)
    maxval = np.abs(Dmax)

    for i in range(0, ntraces, skipt):

       # use copy to avoid truncating the data
        trace = copy.copy(Data[:, i])
        trace = Data[:, i]
        trace[0] = 0
        trace[-1] = 0
        traceplt = x[i] + gain * skipt * dx * trace / maxval
        traceplt = np.clip(traceplt, a_min=x[i]-dx, a_max=(dx+x[i]))

        ax.plot(traceplt, t, color=color, linewidth=lwidth)

        offset = x[i]

        if typeD=='VA':
            for a in range(len(trace)):
                if (trace[a] < 0):
                    trace[a] = 0
            ax.fill_betweenx(t, offset, traceplt, where=(traceplt>offset), interpolate='True', linewidth=0, color=color)
            ax.grid(False)

        ax.set_xlim([x[0]-1, x[-1]+1])

    ax.invert_yaxis()
    ax.set_ylim([np.max(t), np.min(t)])
    ax.set_ylabel(yl)

def plothdr(Header: np.ndarray, trmin=None, trmax=None):

    """
    plothdr(Header, trmin, trmax) - plots the header data

    Parameters
    ----------
    Header: np.ndarray
        Data header

    Optional parameters
    -------------------
    trmin: int
        Start with trace trmin
    trmax: int
        End with trace trmax, int

    Returns
    -------
    Figures:
        four plots:
        (1) shot position
        (2) CMP
        (3) receiver position
        (4) trace offset

    Translated to Python from Matlab by Musab al Hasani and Nicolas Vinard, 2019

    """

    if trmin == None:
        trmin = 0

    if trmax == None:
        trmax = np.max(np.size(Header[0,:]))

    fig, ax = plt.subplots(2,2, figsize=(10,8))

    ind = np.array([2, 4, 3, 1])

    ax[0,0].plot(np.arange(trmin, trmax, 1), Header[1, trmin:trmax], 'x')
    ax[0,1].plot(np.arange(trmin, trmax, 1), Header[3, trmin:trmax], 'x')
    ax[1,0].plot(np.arange(trmin, trmax, 1), Header[2, trmin:trmax], 'x')
    ax[1,1].plot(np.arange(trmin, trmax, 1), Header[4, trmin:trmax], 'x')
    ax[0,0].set(title = 'shot positions', xlabel = 'trace number', ylabel = 'shot position [m]')
    ax[0,1].set(title = 'common midpoint positions (CMPs)', xlabel = 'trace number', ylabel = 'CMP [m]')
    ax[1,0].set(title = 'receiver positions', xlabel = 'trace number', ylabel = 'receiver position [m]')
    ax[1,1].set(title = 'trace offset', xlabel = 'trace number', ylabel = 'offset [m]')
    fig.tight_layout()


##################################################
######### DATA ANALYSIS PRE-PROCESSING ###########
##################################################

def semblanceWiggle(
    CMPgather: np.ndarray,
    H_CMPgather: np.ndarray,
    H,
    vmin:float,
    vmax:float,
    vstep:float
    )->tuple([np.ndarray, np.ndarray]):

    """
    v_picks, t_picks = semblanceWiggle(CMPgather,TrcH,H,vmin,vmax,vstep):
    This funcion generates an interactive semblance plot for the velocity analysis.
    Picks are generated by left-clicking the semblance with the mouse. A red cross
    indicates the location of the picks. Picks can be removed by click in the middle.
    To end the picking press enter.

    Parameters
    ----------
    CMPgather: np.ndarray
        CMP gather
    H_CMPgather: np.ndarray
        CMP header with postional information
    H:
        Seismic data header
    vmin: float
        Minimum velocity in semblance analysis (m/s)
    vmax: float
        Maximum velocity in semblance analysis (m/s)
    vstep: float
        Velocity step between vmin and vmax (m/s)

    Returns
    -------
    v_picks: np.ndarray
        Picked velocities
    t_picks: np.ndarray
        Time picks at picked velocities

    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019
    """

    ExpSmooth=50 # used to be optinal argument

    # Create vectors
    x = H_CMPgather[4,:]
    t = np.arange(0,H['ns']*H['dt']*1e-6, H['dt']*1e-6)
    t2 = np.arange(0,H['ns']*H['dt']*1e-6, H['dt']*1e-6)
    v = np.arange(vmin,vmax+vstep,vstep)

    # Compute the squares
    xsq = np.square(x)
    tsq = np.square(t)
    vsq = np.square(v)

    # reshape for broadcasting
    xsq = xsq.reshape(1,len(xsq),1)
    tsq = tsq.reshape(len(tsq),1,1)
    vsq = vsq.reshape(1,1,len(vsq))

    t = t.reshape((len(t),1,1))
    x = x.reshape((1,len(x),1))
    v = v.reshape((1,1,len(v)))

    T = np.sqrt(tsq+xsq/vsq)

    # Interpolate data to NMO time and stack for each velocity
    tt = np.exp(-ExpSmooth*np.abs(np.subtract(t,t.T)))
    tt = tt.reshape(np.size(t), np.size(t))

    S = np.zeros((np.size(t),np.size(v))) # Preallocate semblance
    q = np.zeros((np.size(t),np.size(x))) # Preallocate temporary container
    b = np.zeros((np.size(t),1))

    for vi in range(0, np.size(v)):

        for j in range(0, np.size(x)):

            q[:,j] = np.interp(T[:,j,vi], t[:,0,0],CMPgather[:,j], left=0,right=0)

        r = np.sum(q,axis=1, keepdims=True)
        C = t*np.size(x)/np.sum(x**2)*x**2 / T
        C[np.isnan(C)] = 0

        # Three expressions for conventional semblance
        Crq = np.sum(tt*np.sum(r*q,axis=1,keepdims=True),axis=0,keepdims=True).T
        Crr = np.sum(tt*np.size(x)*r**2,axis=0,keepdims=True).T
        Cqq = np.sum(tt*np.sum(q**2,axis=1,keepdims=True),axis=0,keepdims=True).T

        normalS = Crq**2/(Crr*Cqq)
        Brq = np.sum(tt*np.sum(r*(C[:,:,vi]*q),axis=1,keepdims=True),axis=0,keepdims=True).T
        Brr = np.sum(tt*np.sum(r**2*C[:,:,vi],axis=1,keepdims=True),axis=0,keepdims=True).T
        Bqq = np.sum(tt*np.sum(C[:,:,vi]*(q**2),axis=1,keepdims=True),axis=0,keepdims=True).T

        # Minimize b
        A = Crr*Bqq+Cqq*Brr
        Rrq=Crq/(Crq-Brq)
        Rrr=Crr/(Crr-Brr)
        Rqq=Cqq/(Cqq-Bqq)

        ind = np.logical_or(
            np.logical_and(
                np.less(Rrr,Rrq), np.less(Rrq,Rqq)),
            np.logical_and(
                np.less(Rqq,Rrq), np.less(Rrq,Rrr)))

        ind = ind.astype(int)

        for ik in range(0, len(ind)-1):
            if ind[ik,0] == 1:
                b[ik] = (1-(2*Crq[ik,0]*Brr[ik,0]*Bqq[ik,0]-Brq[ik,0]*A[ik,0])/(2*Brq[ik,0]*Crr[ik,0]*Cqq[ik,0]-Crq[ik,0]*A[ik,0]))**(-1)
            elif ind[ik,0] == 0:
                b[ik] = Rrq[ik,0]

        ind2 = np.zeros((ind.shape)).astype(int)

        for ij in range(0,len(b)):
            if (b[ij,0] > 1 or b[ij,0] < 0):
                ind2[ij] = 1
                b[ij,0] = 0

            else:
                ind2[ij] = 0

        Wrq = (1-b)*Crq + b*Brq
        Wrr = (1-b)*Crr + b*Brr;
        Wqq = (1-b)*Cqq + b*Bqq;

        tmp = Wrq**2 / (Wrr*Wqq)
        S[:,vi] = tmp.reshape(len(tmp))

        Ind = np.argwhere(ind2)[:,0]
        s = Brq[Ind]**2/(Brr[Ind]*Bqq[Ind]);
        IndF=np.argwhere(Ind)

        IND = []

        for ik in range(0, len(s)):
            if S[ik,vi] > s[ik]:
                IND.append(1)
            else:
                IND.append(0)

        S[IndF[IND],vi]=s[IND]

    fig, ax = plt.subplots(nrows= 1, figsize=(4,10))
    ax.pcolormesh(v, t, S)
    ax.invert_yaxis()
    ax.set_xlabel('Velocity, m/s', fontsize=12)
    ax.set_ylabel('Time, s', fontsize=12)
    ax.set_title('Left-click: pick \n - Middle-click: delete pick \n - Enter: save picks ')
    fig.tight_layout(pad=2.0, h_pad=1.0)
    plt.waitforbuttonpress()
    picks = plt.ginput(-1)
    plt.close()

    picks = np.asarray(picks)
    v_picks = picks[:,0]
    t_picks = picks[:,1]
    index_sort = np.argsort(t_picks)
    t_picks = t_picks[index_sort]
    v_picks = v_picks[index_sort]
    t_picks = np.insert(t_picks, 0, 0)
    v_picks = np.insert(v_picks, 0, v_picks[0])
    t_picks = np.insert(t_picks, len(t_picks), t[-1])
    v_picks = np.insert(v_picks, len(v_picks), v_picks[-1])

    return v_picks, t_picks


def apply_nmo(
    CMPgather: np.ndarray,
    H_CMPgather: np.ndarray,
    H,
    t: np.ndarray,
    v: np.ndarray,
    smute=0
    )->np.ndarray:

    """
    NMOedCMP = apply_nmo(CMPgather, H_CMPgather, H, t, v, smute=0)

    This function applies NMO to a single CMP-gather given a 1D velocity-time log
    Addtionally it outputs three plots showing the log, the CMP-gather before and after NMO

    Parameters
    ----------
    CMPgather gather: np.ndarray
        CMP gather
    H_CMPgather: np.ndarray
        Header of CMP-gather
    H:
        Seismic data header
    t: np.ndarray of shape (#picks,)
        Time of velocity picks in seconds
    v: np.ndarray of shape (#picks,)
        Velocity picks in m/s

    Optinal parameters
    ------------------
    smute: float
        Stretch-mute value (default 0)

    Returns
    -------
    Three plots:
        (1) log
        (2) CMP before
        (3) CMP after
    NMOedCMP: np.ndarray
        NMO-ed CMP gather

    Translated to Python from Matlab by Musab al Hasani and Nicolas Vinard, 2019

    """
    # Convert time to ms
    t = 1000*t
    dt = H['dt']*1e-3 # Convert H['dt'] to ms
    nt = H['ns'] # Number of time samples

    # append zero to first time vector if not already 0
    if t[0] > 0:
        t = np.append(0.0, t)
        v = np.append(v[0], v)

    # End of t should be the total time
    if t[len(t)-1] < dt*nt:
        t = np.append(t, dt*nt)
        v = np.append(v, v[len(v)-1])

    # Plot time-velocity log
    t_plot = np.arange(0,dt*nt, dt)
    v2 = np.interp(t_plot, t, v)

    c = v2
    dt = dt/1000

    nx = np.min(CMPgather.shape)

    NMOedCMP = np.zeros((nt, nx))

    if smute == 0:
        for ix in range(0, nx):

            off = H_CMPgather[4,ix]

            for it in range(0, nt):

                off2c2 = (off*off)/(c[it]*c[it])
                t0 = it * dt
                t2 = t0*t0 + off2c2
                tnmo = np.sqrt(t2) - t0
                itnmo1 = int(np.floor(tnmo/dt))
                difft = (tnmo-dt*itnmo1)/dt

                if (it+itnmo1) < nt:
                    NMOedCMP[it,ix] = (1.-difft)*CMPgather[it+itnmo1,ix] + difft*CMPgather[it+itnmo1,ix]

                if it+itnmo1 == nt:
                    NMOedCMP[it, ix] = CMPgather[it+itnmo1-1, ix]

    else:

        for ix in range(0, nx):

            off = H_CMPgather[4,ix]

            for it in range(0, nt):

                off2c2 = (off*off)/(c[it]*c[it])
                t0 = it * dt
                t02 = t0*t0
                t2 = t02*off2c2
                tnmo = np.sqrt(t2)-t0

                if it==1:
                    dtnmo = 1000.
                else:
                    dtnmo = np.abs(np.sqrt(1+off2c2/t02)) - 1.

                itnmo1 = int(np.floor(tnmo/dt))
                difft = (tnmo-dt*itnmo1)/dt

                if (it+itnmo1) < nt:
                    if dtnmo  >= smute:
                        NMOedCMP[it,ix] = 0.
                    else:
                        NMOedCMP[it, ix] = (1.-difft)*CMPgather[it+itnmo,ix] + difft*CMPgather[it+itnmo1,ix]
                if it+itnmo1 == nt:
                    NMOedCMP[it, ix] = CMPgather[it+itnmo1-1,ix]

    return NMOedCMP

def nmo_v(
    cmp_gather: np.ndarray,
    H_CMPgather: np.ndarray,
    H,
    c: float,
    smute=0
    )->np.ndarray:

    """

    NMOedCMP = nmo_v(cmp_gather, H_CMPgather, H, c, smute=0)

    This function applies NMO to a single CMP-gather, with linear interpolation,
    accroding to a constant velocity c

    Parameters
    ---------
    cmp_gather: np.ndarray
        CMP-gather
    H_CMPgather: np.ndarray
        Header of CMP-gather
    H:
        Data header
    c: float
        Constant velocity in m/s

    Optional parameters
    -------------------
    smute: float
        Stretch-mute value (default 0) means no stretch muting

    Returns
    -------
    NMOedCMP: np.ndarray
        NMO-ed CMP gather

    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    Todo: Change stretch mute default to None or something

    """

    nt, nx = cmp_gather.shape
    dt = H['dt'] / 1000000.
    cmp_new = np.zeros((cmp_gather.shape))

    if smute == 0:

        for ix in range(0, nx):

            off = H_CMPgather[4, ix]
            off2c2 = (off * off)/(c * c)

            for it in range(0, nt):

                t0 = it * dt
                t2 = t0 * t0 + off2c2
                tnmo = np.sqrt(t2) - t0
                itnmo1 = int(np.floor(tnmo / dt))
                difft = (tnmo-dt * itnmo1) / dt

                if it + itnmo1 + 1 < nt:
                    cmp_new[it, ix] = (1. - difft) * cmp_gather[it + itnmo1, ix] + difft * cmp_gather[it + itnmo1 + 1,ix]
                if it+itnmo1 == nt-1:
                    cmp_new[it,ix] = cmp_gather[it+itnmo1,ix]

    else:

        for ix in range(0, nx):

            off    = H_CMPgather[4,ix]
            off2c2 = (off * off)/(c * c)

            for it in range(0, nt):
                t0 = it * dt
                t02 = t0 * t0
                t2 = t02 + off2c2
                tnmo = np.sqrt(t2) - t0

                if it == 0:
                    dtnmo = 1000000.;
                else:
                    dtnmo = np.abs(np.sqrt(1 + off2c2/t02)) - 1.

                itnmo1 = int(np.floor(tnmo / dt))
                difft = (tnmo - dt * itnmo1) / dt

                if it + itnmo1 + 1 < nt:
                    if dtnmo > smute:
                        cmp_new[it, ix] = 0.0
                    else:
                        cmp_new[it, ix] = (1.-difft) * cmp_gather[it + itnmo1, ix] + difft*cmp_gather[it + itnmo1 + 1, ix]

                if it + itnmo1 == nt - 1:
                    cmp_new[it, ix] = cmp_gather[it + itnmo1, ix]

    return cmp_new

def nmo_vlog(
    CMPgather: np.ndarray,
    H_CMPgather: np.ndarray,
    H: dict,
    t: np.ndarray,
    v: np.ndarray,
    smute=0
    )->tuple([np.ndarray, np.ndarray]):

    """
    NMOedCMP = nmo_vlog(CMPgather, H_CMPgather, H, t, v, smute=0)

    This function applied NMO to a single CMP-gather given a 1D velocity-time log
    Addtionally it outputs three plots showing the log, the CMP-gather before and after NMO

    Parameters
    ----------
    CMP gather: np.ndarray
        CMP gather
    H_CMPgather: np.ndarray
        Header of CMP gather
    H: dict
        Seismid data header
    t: np.ndarray of shape (#picks,)
        Time of velocity picks in seconds
    v: np.ndarray of shape (#picks,)
        Velocity picks in m/s

    Optional parameters
    --------------------
    smute:float
        stretch-mute value (default 0) meaning no mute

    Returns
    -------
    Three plots: log, CMO before and after
    NMOedCMP: np.ndarray
        NMO-ed CMP gather
    v_interp: np.ndarray
        Interpolated velocities (same length as time vector)

    Translated to Python from Matlab by Musab al Hasani and Nicolas Vinard, 2019

    """
    # Convert time to ms
    t = 1000*t
    dt = H['dt']*1e-3
    nt = H['ns'] # Number of time samples

    # append zero to first time vector if not already 0
    if t[0] > 0:
        t = np.append(0.0, t)
        v = np.append(v[0], v)

    # End of t should be the total time
    if t[len(t)-1] < dt*nt:
        t = np.append(t, dt*nt)
        v = np.append(v, v[len(v)-1])

    # Plot time-velocity log
    t_plot = np.arange(0,dt*nt, dt)
    v2 = np.interp(t_plot, t, v)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(v2, t_plot)
    ax.scatter(v,t, color='red')
    ax.invert_yaxis()
    ax.set_ylabel('two-way traveltime [ms]')
    ax.set_xlabel('velocity [m/s]')
    fig.tight_layout()

    c = v2
    dt = dt*1e-03 # Now time is needed in s
    nx = np.min(CMPgather.shape)

    NMOedCMP = np.zeros((nt, nx))

    if smute == 0:

        for ix in range(0, nx):

            off = H_CMPgather[4,ix]

            for it in range(0, nt):

                off2c2 = (off*off)/(c[it]*c[it])
                t0 = it * dt
                t2 = t0*t0 + off2c2
                tnmo = np.sqrt(t2) - t0
                itnmo1 = int(np.floor(tnmo/dt))
                difft = (tnmo-dt*itnmo1)/dt

                if (it+itnmo1) < nt:
                    NMOedCMP[it,ix] = (1.-difft)*CMPgather[it+itnmo1-1,ix] + difft*CMPgather[it+itnmo1,ix]
                if it+itnmo1 == nt:
                    NMOedCMP[it, ix] = CMPgather[it+itnmo1-1, ix]

    elif smute!=0:

        for ix in range(0, nx):

            off = H_CMPgather[4,ix]

            for it in range(0, nt):

                off2c2 = (off*off)/(c[it]*c[it])
                t0 = it * dt
                t02 = t0*t0
                t2 = t02 + off2c2
                tnmo = np.sqrt(t2)-t0

                if it==0:
                    dtnmo = 1000.
                else:
                    dtnmo = np.abs(np.sqrt(1+off2c2/t02)) - 1.

                itnmo1 = int(np.floor(tnmo/dt))
                difft = (tnmo-dt*itnmo1)/dt

                if (it+itnmo1) < nt:
                    if dtnmo  >= smute:
                        NMOedCMP[it,ix] = 0.
                    else:
                        NMOedCMP[it, ix] = (1.-difft)*CMPgather[it+itnmo1-1,ix] + difft*CMPgather[it+itnmo1,ix]
                if it+itnmo1 == nt:
                    NMOedCMP[it, ix] = CMPgather[it+itnmo1-1,ix]

    fig, ax = plt.subplots()
    wiggle(CMPgather)
    ax.set_title('Original CMP gather')
    fig.tight_layout()
    fig, ax = plt.subplots()
    wiggle(NMOedCMP)
    ax.set_title('NMO-ed CMP gather')
    fig.tight_layout()

    return NMOedCMP

def nmo_stack(
    cmpsorted_data: np.ndarray,
    cmpsorted_hdr: np.ndarray,
    midpoints: np.ndarray,
    folds: np.ndarray,
    H: dict,
    vmodel: np.ndarray,
    smute=0
    )->np.ndarray:

    #
    zosection = nmo_stack(cmpsorted_data, cmpsorted_hdr, midpoints, folds, H, vmodel, smute=None)

    This function generates a stacked zero-offset section from a CMP-sorted
    dataset. First, NMO correction is performed on each CMP-gather, using the
    velocity model of the subsurface. Subsequently, each NMO'ed CMP-gather is
    stacked to a obtain zero-offset traces on the distances corresponding to
    the midpoints of each CMP-gather.

    Parameters
    ----------
    cmpsorted_data:np.ndarray
        CMP-sorted dataset
    cmpsorted_hdr: np.ndarray
        Its headers
    midpoints: np.ndarray
        CMP-gather positions (see ANALYSEFOLD)
    folds: np.ndarray
        CMP-gather folds (see ANALYSEFOLD)
    H: dict
        Header of the seismic data
    vmodel: np.ndarray
        Velocity model matrix

    Optional parameters
    -------------------
    smute: float
        Stretch-mute factor (default is 0 which equals no mute)

    Returns
    -------
    zosection: np.ndarray
        Zero-offset stacked seismic section

    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019



    # Read the amount of time-samples and traces from the size of the data matrix
    nt,ntr=cmpsorted_data.shape

    # Amount of cmp gathers equals the length of the midpoint-array
    cmpnr = len(midpoints)

    # Initialise tracenr in cmpsorted dataset
    tracenr = 0

    # Initialize zosection
    zosection = np.zeros((nt, cmpnr))

    print('Processing CMPs. This may take some time...')
    print(' ')

    # Update message every tenth percent
    printcounter = 0
    tenPerc = int(cmpnr/10)
    percStatus = 0

    for l in range(0, cmpnr):

        # CMP midpoint in [m] (just for display), and associated fold
        midpoint = midpoints[l]
        fold = folds[l]

        # positioning in the cmpsorted dataset
        gather = cmpsorted_data[:, tracenr:(tracenr+fold)]
        gather_hdr = cmpsorted_hdr[:, tracenr:(tracenr+fold)]

        # NMO and stack the selected CMP-gather
        nmoed = nmo_vxt(gather, gather_hdr, H,  vmodel[:,l], smute)
        zotrace = stack_cmp(nmoed)
        zosection[:,l] = zotrace[:,0]

        # go to traceposition of next CMP in cmpsorted dataset
        tracenr = tracenr + fold

        # Update message
        if printcounter == tenPerc:
            percStatus += 10
            print('Finished stacking {} traces out of {}. {}%'.format(l, cmpnr, percStatus))
            printcounter=0

        printcounter+=1

    print('Done')

    return zosection



"""
def nmo_stack(
    cmpsorted_data: np.ndarray,
    cmpsorted_hdr: np.ndarray,
    midpoints: np.ndarray,
    folds: np.ndarray,
    H: dict,
    vmodel: np.ndarray,
    smute=0
    )->np.ndarray:

    '''
    zosection = nmo_stack(cmpsorted_data, cmpsorted_hdr, midpoints, folds, H, vmodel, smute=None)

    This function generates a stacked zero-offset section from a CMP-sorted
    dataset. First, NMO correction is performed on each CMP-gather, using the
    velocity model of the subsurface. Subsequently, each NMO'ed CMP-gather is
    stacked to a obtain zero-offset traces on the distances corresponding to
    the midpoints of each CMP-gather.

    Parameters
    ----------
    cmpsorted_data:np.ndarray
        CMP-sorted dataset
    cmpsorted_hdr: np.ndarray
        Its headers
    midpoints: np.ndarray
        CMP-gather positions (see ANALYSEFOLD)
    folds: np.ndarray
        CMP-gather folds (see ANALYSEFOLD)
    H: dict
        Header of the seismic data
    vmodel: np.ndarray
        Velocity model matrix

    Optional parameters
    -------------------
    smute: float
        Stretch-mute factor (default is 0 which equals no mute)

    Returns
    -------
    zosection: np.ndarray
        Zero-offset stacked seismic section

    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    '''

    # Read the amount of time-samples and traces from the size of the datamatrix
    nt,ntr=cmpsorted_data.shape

    # Amount of cmp gathers equals the length of the midpoint-array
    cmpnr = len(midpoints)

    # Initialise tracenr in cmpsorted dataset
    trace_num = 1
    zosection = np.zeros((nt, cmpnr))

    for l in tnrange(cmpnr, desc='Processing CMPs'):
        # CMP midpoint in [m] (just for display), and associated fold
        midpoint = midpoints[l]
        fold = folds[l]

        # positioning in the cmpsorted dataset
        gather = cmpsorted_data[:, (trace_num-1):(trace_num+fold)]
        gather_hdr = cmpsorted_hdr[:, (trace_num-1):(trace_num+fold)]

        # NMO and stack the selected CMP-gather
        nmoed = nmo_vxt(gather, gather_hdr, H,  vmodel[:,l], smute)
        zotrace = stack_cmp(nmoed)
        zosection[:,l] = zotrace[:,0]

        # go to traceposition of next CMP in cmpsorted dataset
        trace_num = trace_num + fold

    return zosection

"""

def stackplot(gather: np.ndarray, H: dict)->np.ndarray:

    """
    stack = stackplot(gather, H)

    This function stacks the traces in a gather and makes a plot of the stacked trace

    Parameters
    ----------
    gather: np.ndarray
        The gather to stack, usually an NMO'ed CMP gather
    H: dict
        Seimic data header

    Returns
    -------
    stack: np.ndarray
        Stacked trace

    Translated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    """

    gathersize = gather.shape[1]
    stack = np.sum(gather,1)/gathersize
    t = np.arange(0, H['ns']*H['dt']/1000000, H['dt']/1000000)
    d = 0.0

    fig, ax = plt.subplots(figsize=(3,10))
    ax.plot(stack, t, color='green')
    ax.set_title('stacked trace', fontweight='bold')
    ax.set_ylabel('time [s]')
    ax.set_xlabel('amplitude')
    ax.fill_betweenx(t,d,stack, where=(stack>d), color='green')
    ax.invert_yaxis()
    fig.tight_layout()

    stack = stack.reshape(len(stack), 1)

    return stack

def nmo_vxt(
    CMPgather: np.ndarray,
    H_CMPgather: np.ndarray,
    H: dict,
    c: float,
    smute=0
    )-> np.ndarray:

    """

    Parameters
    ----------
    CMPgather: np.ndarray
        CMP gather
    H_CMPgather: np.ndarray
        Its header
    H: dict
        Seismic data header
    c: float
        Velocity in m/s
    smute: float
        Stretch mute parameters, optional

    Returns
    -------
    NMOedCMP: np.ndarray
        NMO-ed CMP gather
    """

    dt = H['dt']*1e-6 # time in seconds
    nt = H['ns'] # Number of time samples
    nx = np.min(CMPgather.shape)

    NMOedCMP = np.zeros((nt, nx))

    if smute == 0:

        for ix in range(0, nx):

            off = H_CMPgather[4,ix]

            for it in range(0, nt):

                off2c2 = (off*off)/(c[it]*c[it])
                t0 = it * dt
                t2 = t0*t0 + off2c2
                tnmo = np.sqrt(t2) - t0
                itnmo1 = int(np.floor(tnmo/dt))
                difft = (tnmo-dt*itnmo1)/dt

                if (it+itnmo1) < nt:
                    NMOedCMP[it,ix] = (1.-difft)*CMPgather[it+itnmo1,ix] + difft*CMPgather[it+itnmo1,ix]

                if it+itnmo1 == nt:
                    NMOedCMP[it, ix] = CMPgather[it+itnmo1-1, ix]

    else:

        for ix in range(0, nx):

            off = H_CMPgather[4,ix]

            for it in range(0, nt):

                off2c2 = (off*off)/(c[it]*c[it])
                t0 = it * dt
                t02 = t0*t0
                t2 = t02 + off2c2
                tnmo = np.sqrt(t2)-t0

                if it == 0:
                    dtnmo = 1000.
                else:
                    dtnmo = np.abs(np.sqrt(1+off2c2/t02)) - 1.

                itnmo1 = int(np.floor(tnmo/dt))
                difft = (tnmo-dt*itnmo1)/dt

                if (it+itnmo1) < nt:
                    if dtnmo  >= smute:
                        NMOedCMP[it,ix] = 0.
                    else:
                        NMOedCMP[it, ix] = (1.-difft)*CMPgather[it+itnmo1-1,ix] + difft*CMPgather[it+itnmo1,ix]

                if it+itnmo1 == nt:
                    NMOedCMP[it, ix] = CMPgather[it+itnmo1-1,ix]

    return NMOedCMP

def stack_cmp(gather: np.ndarray)->np.ndarray:

    """

    stacked_trace = stack_cmp(gather)

    This function stacks one NMO-ed CMP-gather. Output is one stacked trace
    for the midpoint position belonging to the CMP-gather. Is used by the
    function NMO_STACK, use STACKPLOT instead.

    Translated to Python from Matlab by Musab al Hasani and Nicolas Vinard, 2019

    """

    cmpsize = np.min(gather.shape)

    if cmpsize > 1:
        stacked_trace = np.sum(gather,axis=1)
    else:
        stacked_trace = gather

    stacked_trace=np.reshape(stacked_trace,(len(stacked_trace),1))

    return stacked_trace

def generatevmodel2(
    cmppicks: list([np.ndarray]),
    tvpicks: list([[np.ndarray],[np.ndarray]]),
    midpnts: np.ndarray,
    H: dict
    )->np.ndarray:

    """

    vmodel = generatevmodel2(cmppicks, tvpicks, midpnts, H)

    This function generates 2-D velocity model given cmp location, traveltimes,
    Header and midpoints. Linear interpolation between inputs.

    Parameters
    ----------
    cmppicks: list([np.ndarray])
        Picked common midpoints, list([cmp positions])
    tvpicks: list([[np.ndarray],[np.ndarray]])
        Picked traveltimes
    midpnts: np.ndarray
        Midpoint positons (returned by analysefold)
    H: dict
        Seismic data header

    Returns
    -------
    vmodel: np.ndarray
        2-D velocity model

    Written by Nicolas Vinard and Musab al Hasani, 2019

    """

    tvLog = tvLogs(cmppicks, tvpicks, H)
    t = tvLog[:,:,0]*1000
    v = tvLog[:,:,1]
    cmppicks = np.array(cmppicks)

    # Calculating CMP sequence numbers [] from midpoint positions [m]
    #Note that midpnts is an input argument from generatevmodel, the function calling this script
    cmpdist = midpnts[1]-midpnts[0]
    cmp_initoffset = midpnts[0]/cmpdist;
    vcmp = []

    for a in range(0, len(cmppicks)):
        vcmp.append(cmppicks[a]/cmpdist - (cmp_initoffset-1))

    # Preparing the velocity and time matrices for generatevmodel
    nrows, ncols = t.shape
    t_max=H['dt']/1000*(H['ns']-1) # in ms

    t_up = t
    v_up = v

    # Loop over rows, the amount of CMPS
    for k in range(0, len(cmppicks)):

        if t[k,0] > 0:
            t_up[k,:] = np.insert(t_up, 0, 0)
            v_up[k,:] = np.insert(v_up, 0, v[k,0])
        else:
            if k == 0:
                t_up = np.insert(t_up, 1, H['dt']/1000,axis=1)
                v_up = np.insert(v_up, 1, v[k,1], axis=1)
            else:
                t_up[k,0] = t[k,0]
                v_up[k,:1] = v[k,:1]

    t = t_up.astype(dtype='int32')*int(1000)
    v = v_up.astype(dtype='int32')

    # count CMP midpoints
    m=len(midpnts)

    # initialise the vmodel, the columns contain the velocities for each CMP midpnt
    vxt = np.zeros((H['ns'],m))
    vmodel = np.zeros((H['ns'],m))
    vcmp_extended = np.insert(vcmp, 0, 0)
    vcmp_extended = np.append(vcmp_extended, m-1)

    v_extended = v
    t_extended = t
    v_extended = np.append(v_extended, [v[v.shape[0]-1,:]], axis=0)
    t_extended = np.append(t_extended, [t[t.shape[0]-1,:]], axis=0)
    v_extended = np.insert(v_extended, 0, v[0,:], axis=0)
    t_extended = np.insert(t_extended, 0, t[0,:], axis=0)

    vold = np.zeros((H['ns'],np.size(vcmp_extended)))

    for r in range(0, H['ns']):

        for j, k in enumerate(vcmp_extended):

            vxt[:,int(k)] = vlog_exd(v_extended[j,:],t_extended[j,:],H)
            vold[r,j]=vxt[r,int(k)]

        # horizontal interpolation between the picked CMP positions
        x = vcmp_extended
        x2 = np.arange(0,m)
        v2 = np.interp(x2, x, vold[r,:])
        vmodel[r,:] = v2

    # Plot velocity model
    plt.figure()
    plt.pcolormesh(midpnts, np.arange(0,H['dt']*H['ns'],H['dt']), vmodel)
    plt.xlabel('CMP position [m]')
    plt.ylabel('two-way time [ms]')
    plt.title('velocity model')
    plt.gca().invert_yaxis()
    plt.colorbar();

    return vmodel


### Helper functions
def tvLogs(
    cmppicks: list([np.ndarray]),
    tvpicks: list([[np.ndarray], [np.ndarray]]),
    H: dict
    )->np.ndarray:

    """
    tvLog = tvLogs(cmppicks, tvpicks, H)

    This function interpolates the picks along the time dimension

    Written by Nicolas Vinard, 2019

    """

    cmppicks=np.array(cmppicks)
    time = np.arange(0, H['ns'])*H['dt']*1e-06
    tvLog = np.zeros((cmppicks.shape[0], time.shape[0], 2))
    tvLog[:,:,0] = time

    for i in range(len(tvpicks[0])):
        vInterp = np.interp(time, np.array(tvpicks[0][i]), np.array(tvpicks[1][i]))
        tvLog[i,:,1] = vInterp

    return tvLog


def vlog_exd(v:np.ndarray,t:np.ndarray,H:dict):
    """
    Interpolation function used in genereatevmodel2
    """

    dt=H['dt']
    nt=H['ns']
    t2 = np.arange(0,nt*dt,dt)

    return np.interp(t2,t,v)

def vel_zeroOffset(xs, x1, x2, t1, t2):

    """
    Compute velocity estimate of zero offset hyperbola
    """

    velz0 = 2./np.sqrt( np.abs(t2**2 - t1**2) ) * np.sqrt( np.abs( (x2-xs)**2 - (x1-xs)**2 ) )

    return velz0

'''
# kirk_mig with fancy update toolbar. uncomment if you want to use it and then comment the other kirk_mig function
def kirk_mig(dataIn, vModel, t, x):

    """
    dataMig, tmig, xmig = kirk_mig(dataIn, vModel, t, x)

    This functions performs Kirchhoff time migration.

    Parameters
    ----------
    dataIn: np.ndarray
        Zero offset data. One trace per column.
    vModel: float, np.ndarray (1D), np.ndarray (2D)
        Velocity model. Can be in three formats:
            1) float --> constant velocity migration
            2) 1-D np.ndarray --> must have same dimension as the number rows in dataIn.
            In this case it is assumed to be an rms velocity function (of time)
            which is applied at all positions along the section.
            3) 2-D array --> must have same shape as dataIn. Here it is assumed
            to be the rms velocity for each sample location.

    t: float or np.ndarray
        Time information. Two possibilies:
            (1) scalar: time sample rate in seconds
            (2) 1-D np.ndarray: time coordinates for the rows of dataIn.

    x: float or np.ndarray
        Spatial information. Two possibilities:
            (1) float: spatial sample rate (in units consistent with the velocity information.
            (2) 1-D np.ndarray: x-coordinates of the columns of dataIn

    Returns
    -------

    dataMig: np.ndarray
        The output migrated time section
    tmig: np.ndarray
        Time coordinates of migrated data
    xmig: np.ndarray
        Spatial coordinates of migrated data in x

    Tranlated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019
    """

    if np.size(vModel) == 1:
        vModel = np.array(vModel)
        vModel.shape = (1,1)

    nsamp, ntr = dataIn.shape
    nvsamp, nvtr = vModel.shape

    dx = x[1]-x[0]
    dt = t[1]-t[0]

    #  ---- test velocity info ----
    if(nvsamp==1 and nvtr!=1):
        # might be transposed vector
        if(nvtr==nsamp):
            vModel=vModel.T
        else:
            print('Velocity vector is wrong size')
            sys.exit()

        # make velocity matrix
        vModel=vModel*np.ones((1,ntr))

    elif( nvsamp==1 and nvtr==1):
        vModel=vModel*np.ones((nsamp,ntr))
    elif( nvsamp==nsamp and nvtr==1):
        vModel=vModel*np.ones((1,ntr))
    else:
        if(nvsamp!=nsamp):
            print('Velocity matrix has wrong number of rows')
            sys.exit()

        elif(ntr!=nvtr):
            print('Velocity matrix has wrong number of columns')
            sys.exit()

    # Now velocity matrix is of same size as data matrix
    aper = np.abs(np.max(x)-np.min(x))
    width1 = aper/20
    itaper1 = 1
    ang_limit = np.pi/3
    width2 = 0.15*ang_limit
    angle1 = ang_limit + width2
    itaper2 = 1
    interp_type = 1
    tmig1 = np.min(t)
    tmig2 = np.max(t)
    xmig1 = np.min(x)
    xmig2 = np.max(x)
    ibcfilter = 0

    # Aperature in traces and the taper coefficient
    traper0 = 0.5*aper/dx
    traper1 = width1/dx
    traper = np.asarray(np.round(traper0+traper1), dtype='int')
    coef1 = cos_taper(traper0,traper0+traper1)

    # one way time
    dt1 = 0.5*dt
    t1 = t/2.
    t2 = np.power(t1,2)

    # compute maximum time needed
    vmin = np.min(vModel)
    tmax = np.sqrt( 0.25*tmig2**2 + ((0.5*aper+width1)/vmin)**2)

    # pad input to tmaxin
    npad=np.ceil(tmax/dt1)-nsamp+5

    if npad > 0:
        npad2 = ((0, int(npad)), (0,0))
        dataIn = np.pad(dataIn, pad_width=npad2, mode='constant', constant_values=0)
        t1 = np.append(t1, np.arange(nsamp,nsamp+npad)*dt1)

    # output samples targeted ! HERE WE TAKE THE ENTIRE INPUT TIME
    t1.shape = (len(t1),1)
    tmig=t
    samptarget=np.arange(0, len(t))

    # output traces desired ! HERE WE TAKE ALL THE TRACES GIVEN OTHERWISE SEE ORIGINAL CREWES CODE AND ADAPT CODE
    trtarget = np.arange(0, len(x))
    xmig=x

    # initialize output array
    dataMig=np.zeros((len(samptarget), len(trtarget)))

    #loop over migrated traces
    kmig=0
    print(' ')
    print(' --- Total number of traces to be migrated : ' + np.str(len(xmig)) + ' --- ')
    print(' ')

    printcounter = 0
    tenPerc = int(len(trtarget)/10)
    percStatus = 0

    for ktr in tnrange(len(trtarget), desc='Migrating data'):

        ktr2 = trtarget[ktr]

        # determine traces in aperture
        n1=np.max((0, ktr-traper))
        n2=np.min((ntr, ktr+traper))
        truse = np.arange(n1,n2)

        # offsets and velocity
        offset2=np.power(((truse-ktr)*dx),2)
        v2 = np.power(vModel[:,ktr],2)

        for kaper in range(0, len(truse)):

            # offset times
            t_aper = np.sqrt(np.divide(offset2[kaper],v2[samptarget]) + t2[samptarget])

            # cosine theta amplitude correction
            if truse[kaper] == ktr:
                costheta = np.ones(samptarget.shape)
                tanalpha = np.zeros(samptarget.shape)
            else:
                costheta = 0.5*np.divide(tmig,t_aper)
                tanalpha = np.sqrt(1-np.power(costheta,2))

                # angle limit and the taper
                ind = np.where( costheta < np.cos(angle1))[0]
                i1 = ind[len(ind)-1]
                ind = np.where( costheta < np.cos(ang_limit))[0]
                i2 = ind[len(ind)-1]

                if i1 < i2:
                    coef2 = cos_taper(i2,i1)
                    costheta[0:i1+1] = np.zeros((i1+1))
                    costheta[i1+1:i2+1] = np.multiply( np.flip(coef2,axis=0)[i2-i1:],costheta[i1+1:i2+1])

            tmp0 = dataIn[:,truse[kaper]]

            # Linear interpolation ONLY OPTION FOR NOW. CAN BE EXTENDED TO OTHER SCHEMES IF NECESSARY
            tnumber = t_aper/dt1
            it0 = np.array(np.floor( tnumber ), dtype='int')
            it1 = np.array(it0+1,dtype='int')
            xt0 = np.array(tnumber - it0+1, dtype='int')
            xt1 = np.array(it0-tnumber, dtype='int')
            tmp = np.multiply(xt1,tmp0[it0])+np.multiply(xt0, tmp0[it1])

            # aperture taper
            ccoef = 1.

            if np.abs(truse[kaper]-ktr)*dx > 0.5*aper:
                ccoef = coef1[int(np.round(np.abs(truse[kaper]-ktr)-traper0-1))]
            if np.abs(1-ccoef) > 0.05:
                tmp = np.multiply(tmp, ccoef)

            ind = np.where( costheta < 0.999)[0]
            costheta[ind] = np.sqrt(np.power(costheta[ind],3))
            tmp[ind] = np.multiply(tmp[ind], costheta[ind])
            dataMig[:,kmig] = dataMig[:,kmig]+tmp

        # scaling and 45 degree phase shift
        scalemig = np.multiply(vModel[samptarget,kmig],np.sqrt(np.multiply(np.pi,(tmig+0.0001))))
        dataMig[:,kmig] = np.divide(dataMig[:,kmig],scalemig)
        kmig+=1 # numerator

    # 45 degree phase shift
    dataMig = conv45(dataMig)

    return dataMig, tmig, xmig

'''

def cos_taper(sp,ep,samp=1):

    dd=[]
    sp = sp+1
    ep = ep+1
    l = np.abs(ep-sp)/samp
    l = l+1

    if l <= 1:
        coef = np.asarray([1.0])
    if l > 1:
        coef = np.zeros(int(l))
        dd = 1.0/(l-1)*np.pi*0.5

        for i in range(0,int(l)):
            coef[i] = np.cos((i)*dd)

    return coef

def conv45(dataIn):

    itrans = 0
    nrow,nvol=dataIn.shape

    if nrow == 1:
        dataIn=dataIn.T
        nrow = nvol
        nvol = 1
        itrans = 1

    aryout=np.zeros(dataIn.shape)
    filt = np.array([-0.0010 -0.0030,-0.0066,-0.0085,-0.0060, -0.0083, -0.0107,
            -0.0164,-0.0103,-0.0194,-0.0221,-0.0705,0.0395,-0.2161,-0.3831,
            0.5451,0.4775,-0.1570,0.0130,0.0321,-0.0129]).T

    for j in range(0,nvol):
        conv1=np.convolve(dataIn[:,j], filt)
        aryout[:,j]=conv1[15:nrow+15]

    if itrans is True:
        aryout = aryout.T

    return aryout


def kirk_mig(
    dataIn: np.ndarray,
    vModel,
    t,
    x
    )->tuple([np.ndarray, np.ndarray, np.ndarray]):

    """
    dataMig, tmig, xmig = kirk_mig(dataIn, vModel, t, x)

    This functions performs Kirchhoff time migration.

    Parameters
    ----------
    dataIn: np.ndarray
        Zero offset data. One trace per column.
    vModel: float, np.ndarray (1D), np.ndarray (2D)
        Velocity model. Can be in three formats:
            1) float --> constant velocity migration
            2) 1-D np.ndarray --> must have same dimension as the number rows in dataIn.
            In this case it is assumed to be an rms velocity function (of time)
            which is applied at all positions along the section.
            3) 2-D array --> must have same shape as dataIn. Here it is assumed
            to be the rms velocity for each sample location.

    t: float or np.ndarray
        Time information. Two possibilies:
            (1) scalar: time sample rate in seconds
            (2) 1-D np.ndarray: time coordinates for the rows of dataIn.

    x: float or np.ndarray
        Spatial information. Two possibilities:
            (1) float: spatial sample rate (in units consistent with the velocity information.
            (2) 1-D np.ndarray: x-coordinates of the columns of dataIn

    Returns
    -------

    dataMig: np.ndarray
        The output migrated time section
    tmig: np.ndarray
        Time coordinates of migrated data
    xmig: np.ndarray
        Spatial coordinates of migrated data in x

    Tranlated to Python from Matlab by Nicolas Vinard and Musab al Hasani, 2019

    """

    if np.size(vModel) == 1:
        vModel = np.array(vModel)
        vModel.shape = (1,1)

    nsamp, ntr = dataIn.shape
    nvsamp, nvtr = vModel.shape

    dx = x[1]-x[0]
    dt = t[1]-t[0]

    #  ---- test velocity info ----
    if(nvsamp==1 and nvtr!=1):
        # might be transposed vector
        if(nvtr==nsamp):
            vModel=vModel.T
        else:
            print('Velocity vector is wrong size')
            sys.exit()

        # make velocity matrix
        vModel=vModel*np.ones((1,ntr))

    elif( nvsamp==1 and nvtr==1):
        vModel=vModel*np.ones((nsamp,ntr))
    elif( nvsamp==nsamp and nvtr==1):
        vModel=vModel*np.ones((1,ntr))
    else:
        if(nvsamp!=nsamp):
            print('Velocity matrix has wrong number of rows')
            sys.exit()
        elif(ntr!=nvtr):
            print('Velocity matrix has wrong number of columns')
            sys.exit()

    # Now velocity matrix is of same size as data matrix
    aper = np.abs(np.max(x)-np.min(x))
    width1 = aper/20
    itaper1 = 1
    ang_limit = np.pi/3
    width2 = 0.15*ang_limit
    angle1 = ang_limit + width2
    itaper2 = 1
    interp_type = 1
    tmig1 = np.min(t)
    tmig2 = np.max(t)
    xmig1 = np.min(x)
    xmig2 = np.max(x)
    ibcfilter = 0

    # Aperature in traces and the taper coefficient
    traper0 = 0.5*aper/dx
    traper1 = width1/dx
    traper = np.asarray(np.round(traper0+traper1), dtype='int')
    coef1 = cos_taper(traper0,traper0+traper1)

    # one way time
    dt1 = 0.5*dt
    t1 = t/2.
    t2 = np.power(t1,2)

    # compute maximum time needed
    vmin = np.min(vModel)
    tmax = np.sqrt( 0.25*tmig2**2 + ((0.5*aper+width1)/vmin)**2)

    # pad input to tmaxin
    npad=np.ceil(tmax/dt1)-nsamp+5
    if npad > 0:
        npad2 = ((0, int(npad)), (0,0))
        dataIn = np.pad(dataIn, pad_width=npad2, mode='constant', constant_values=0)
        t1 = np.append(t1, np.arange(nsamp,nsamp+npad)*dt1)

    # output samples targeted ! HERE WE TAKE THE ENTIRE INPUT TIME
    t1.shape = (len(t1),1)
    tmig=t
    samptarget=np.arange(0, len(t))

    # output traces desired ! HERE WE TAKE ALL THE TRACES GIVEN OTHERWISE SEE ORIGINAL CREWES CODE AND ADAPT CODE
    trtarget = np.arange(0, len(x))
    xmig=x

    # initialize output array
    dataMig=np.zeros((len(samptarget), len(trtarget)))

    #loop over migrated traces
    kmig=0
    print(' ')
    print(' --- Total number of traces to be migrated : ' + np.str(len(xmig)) + ' --- ')
    print(' ')

    printcounter = 0
    tenPerc = int(len(trtarget)/10)
    percStatus = 0

    for ktr, ktr2 in enumerate(trtarget):  # ktr - location of output trace

        # determine traces in aperture
        n1=np.max((0, ktr-traper))
        n2=np.min((ntr, ktr+traper))
        truse = np.arange(n1,n2)

        # offsets and velocity
        offset2=np.power(((truse-ktr)*dx),2)
        v2 = np.power(vModel[:,ktr],2)

        # loop over traces in aperture
        for kaper in range(0, len(truse)):

            # offset times
            t_aper = np.sqrt(np.divide(offset2[kaper],v2[samptarget]) + t2[samptarget])

            # cosine theta amplitude correction
            if truse[kaper] == ktr:
                costheta = np.ones(samptarget.shape)
                tanalpha = np.zeros(samptarget.shape)
            else:
                costheta = 0.5*np.divide(tmig,t_aper)
                tanalpha = np.sqrt(1-np.power(costheta,2))

                # angle limit and the taper
                ind = np.where( costheta < np.cos(angle1))[0]
                i1 = ind[len(ind)-1]
                ind = np.where( costheta < np.cos(ang_limit))[0]
                i2 = ind[len(ind)-1]

                if i1 < i2:
                    coef2 = cos_taper(i2,i1)
                    costheta[0:i1+1] = np.zeros((i1+1))
                    costheta[i1+1:i2+1] = np.multiply( np.flip(coef2,axis=0)[i2-i1:],costheta[i1+1:i2+1])

            tmp0 = dataIn[:,truse[kaper]]

            # Linear interpolation ONLY OPTION FOR NOW. CAN BE EXTENDED TO OTHER SCHEMES IF NECESSARY
            tnumber = t_aper/dt1
            it0 = np.array(np.floor( tnumber ), dtype='int')
            it1 = np.array(it0+1,dtype='int')
            xt0 = np.array(tnumber - it0+1, dtype='int')
            xt1 = np.array(it0-tnumber, dtype='int')
            tmp = np.multiply(xt1,tmp0[it0])+np.multiply(xt0, tmp0[it1])

            # aperture taper
            ccoef = 1.
            if np.abs(truse[kaper]-ktr)*dx > 0.5*aper:
                ccoef = coef1[int(np.round(np.abs(truse[kaper]-ktr)-traper0-1))]
            if np.abs(1-ccoef) > 0.05:
                tmp = np.multiply(tmp, ccoef)

            ind = np.where( costheta < 0.999)[0]
            costheta[ind] = np.sqrt(np.power(costheta[ind],3))
            tmp[ind] = np.multiply(tmp[ind], costheta[ind])
            dataMig[:,kmig] = dataMig[:,kmig]+tmp

        # scaling and 45 degree phase shift
        scalemig = np.multiply(vModel[samptarget,kmig],np.sqrt(np.multiply(np.pi,(tmig+0.0001))))
        dataMig[:,kmig] = np.divide(dataMig[:,kmig],scalemig)
        kmig+=1 # numerator

        # Print progress information
        if printcounter == tenPerc:
            percStatus += 10
            print('Finished migrating {} traces out of {}. {}%'.format(ktr, len(trtarget), percStatus))
            printcounter=0

        printcounter+=1

    # 45 degree phase shift
    dataMig = conv45(dataMig)
    print('Done')

    return dataMig, tmig, xmig


def time2depth_trace(ttrace, vrmsmodel, tt):

    """
    time2depth: Convert a  single trace in the time domain to the depth domaing
    using a RMS-velocity model

     Usage:
         [ztrace,zz]=time2depth_trace(ttrace,vrmsmodel,tt)

     Output:  ztrace    - depth-converted trace
              zz        - depth vector

     Input:   ttrace    - trace in time domain
              vmodel    - 1D RMS-velocity model
              tt        - time vector


    TIME2DEPTH_TRACE is a Matlab function originally written by Guy Drijkoningen
    and translated to Python by Musab Al Hasani.

    """

    dt = tt[1] - tt[0]
    nt = len(tt)

    vintmodel    =  np.zeros(nt)
    vintmodel[0] =  vrmsmodel[0]

    for it in range(1, nt):
        v2diff = tt[it]*vrmsmodel[it]**2 - tt[it-1]*vrmsmodel[it-1]**2
        vintmodel[it] = np.sqrt(v2diff/dt)

    # determine minumum velocity for minimum sampling in depth z
    vrmsmin = np.min(vrmsmodel)

    # take dz as smallest velocity times dt/2 (two-way time):
    dz = vrmsmin*dt/2

    # take maximum depth as maximum velocity times tmax/2 (two-way time):
    tmax = tt[-1]
    vrmsmax = np.max(vrmsmodel)
    zmax = vrmsmax*tmax/2

    nz = int(np.ceil(zmax/dz+1))
    zmax2 = nz*dz
    zz = np.arange(dz, zmax2, dz)

    # now we need to interpolate to regulaa np.range(dz, zmax) with dz step

    ztrace = np.zeros((nz))
    ztrace[0]      = ttrace[0]
    itrun          = 0
    z1             = 0.0
    z2             = zmax2

    for iz in range(1,int(nz)):

        ztrue = iz*dz

        # find out between which time samples are needed for interpolation:
        if itrun < nt:
            z2 = z1 + (vintmodel[itrun-1]*dt/2)
            while ztrue > z2 and itrun < nt:

                itrun = itrun +1
                z1    = z2
                z2    = z2 + vintmodel[itrun-1]*dt/2

            if itrun < nt:
                ztrace[iz] = (z2-ztrue)/(z2-z1)*ttrace[itrun-1] + (ztrue-z1)/(z2-z1)*ttrace[itrun]

    print('Done!')

    return ztrace, zz





def time2depth_section(tsection, vrmsmodel, tt):

    """

    time2depth: Convert a  time-migrated section to a depth
     section, using a RMS-velocity model

     Usage:
         [zmigsection,zz]=time2depth_SECTION(tmigsection,vrmsmodel,tt)

     Output:  zsection  - depth-converted time-migrated section
              zz        - depth vector

     Input:   tsection  - time (possibly time-migrated) section
              vmodel    - RMS-velocity model
              tt        - time vector


    TIME2DEPTH_SECTION is a Matlab function originally written by Guy Drijkoningen
    and translated to Python by Musab Al Hasani.

    """

    dt = tt[1] - tt[0]
    nt = len(tt)
    nx = tsection.shape[1]

    vintmodel       = np.zeros((nt,nx))
    ix              = 0
    vintmodel[0,ix] =  vrmsmodel[0,ix]

    for it in range(1, nt):
        v2diff = tt[it]*vrmsmodel[it, ix]**2 - tt[it-1]*vrmsmodel[it-1, ix]**2
        vintmodel[it, ix] = np.sqrt(v2diff/dt)

    for ix in range(1,nx):
        vintmodel[0,ix] = vrmsmodel[0,ix]
        for it in range(1,nt):
            v2diff = tt[it]*vrmsmodel[it,ix]**2 - tt[it-1]*vrmsmodel[it-1,ix]**2
            vintmodel[it,ix] = np.sqrt(v2diff/dt)

    # determine minumum velocity for minimum sampling in depth z
    vrmsmin = np.min(vrmsmodel)

    # take dz as smallest velocity times dt/2 (two-way time):
    dz = vrmsmin*dt/2

    # take maximum depth as maximum velocity times tmax/2 (two-way time):
    tmax = tt[len(tt)-1]
    vrmsmax = np.max(vrmsmodel)
    zmax = vrmsmax*tmax/2
    nz = int(np.ceil(zmax/dz+1))
    zmax2 = nz*dz
    zz = np.arange(dz, zmax2+dz, dz)

    print(' ')
    print(' --- Total number of traces to be converted to depth: ' + np.str(nx) + ' --- ')
    print(' ')

    # now we need to interpolate to regulaa np.range(dz, zmax) with dz step
    zsection = np.zeros((nz, nx))

    printcounter = 0
    tenPerc = int(nx/10)
    percStatus = 0

    for ix in range(0,nx):

        zsection[0,ix] = tsection[0,ix]
        itrun          = 0
        z1             = 0.0
        z2             = zmax2

        for iz in range(1,int(nz)):

            ztrue = iz*dz

            # find out between which time samples are needed for interpolation:
            if itrun < nt:
                z2 = z1 + (vintmodel[itrun-1, ix]*dt/2)
                while ztrue > z2 and itrun < nt:
                    itrun = itrun +1
                    z1    = z2
                    z2    = z2 + vintmodel[itrun-1,ix]*dt/2

                if itrun < nt:
                    zsection [iz, ix] = (z2-ztrue)/(z2-z1)*tsection[itrun-1, ix] + (ztrue-z1)/(z2-z1)*tsection[itrun, ix]

        if printcounter == tenPerc:
            percStatus += 10
            print('Finished depth converting {} traces out of {}. {}%'.format(ix, nx, percStatus))
            printcounter=0

        printcounter+=1

    print('Done!')

    return zsection, zz


def agc(DataO: np.ndarray, time: np.ndarray, agc_type = 'inst',  time_gate = 500e-3):
    """
    agc: applies automatic gain control for a given dataset.

     Usage:
         gained_data = agc(data,time,agc_type, time_gate)

     Parameters
     -----------
     data: np.ndarray
            Input seismic data
     time: np.ndarray
            Time array
     agc_type: string <class 'str'>
            Type of agc to be applied. Options: 1)'inst': instantanous AGC. 2) 'rms': root-mean-square.
            For details, please refere to: https://wiki.seg.org/wiki/Gain_applications
     time_gate: float <class 'float'>
            Time gate used for agc in sec. Defualt value 500e-3.

     Returns
     -------
     gained_data: np.ndarray
        Data after applying AGC

        AGC is python function written by Musab Al Hasani based on the book of Oz Yilmaz (https://wiki.seg.org/wiki/Gain_applications)

    """
    data = np.copy(DataO)

    # # calculate nth-percentile
    # nth_percentile = np.abs(np.percentile(data, 99))

    # clip data to the value of nth-percentile
    # data = np.clip(data, a_min=-nth_percentile, a_max = nth_percentile)


    num_traces = data.shape[1] # number of traces to apply gain on
    gain_data  = np.zeros(data.shape) # initialise the gained data 2D array

    # check what type of agc to use
    if agc_type == 'rms':
        for itrc in range(num_traces):
            gain_data[:, itrc] = rms_agc(data[:, itrc], time, time_gate)

    elif agc_type =='inst':
        for itrc in range(num_traces):
            gain_data[:, itrc] = inst_agc(data[:, itrc], time, time_gate)

    else:
        print('Wrong agc type!')

    return gain_data



def rms_agc(trace: np.ndarray, time: np.ndarray,  time_gate=200e-3)-> np.ndarray:
    """

    rms_agc: apply root-mean-square automatic gain control for a given trace.

     Usage:
         gained_trace = agc(data,time,agc_type, time_gate)

     Parameters
     -----------
     data: np.ndarray
            Input seismic trace
     time: np.ndarray
            Time array
     time_gate: float <class 'float'>
            Time gate used for agc in sec. Defualt value 200e-3 here, though there is  not a typecal value to be used.

     Returns
     -------
     gained_trace: np.ndarray
        trace after applying RMS AGC

        RMS_AGC is python function written by Musab Al Hasani based on the book of Oz Yilmaz (https://wiki.seg.org/wiki/Gain_applications)

    """

    # determine time sampling and num of samples
    dt = time[1]-time[0]
    N = len(trace)

    # determine number of time gates to use
    gates_num = int((time[-1]//time_gate)+1)

    # initialise indecies for the coners of the gate
    time_gate_1st_ind = 0
    time_gate_2nd_ind = int(time_gate/dt)


    # construct lists for begining and ends of tome gates
    start_gate_inds = [(time_gate_1st_ind + i*time_gate_2nd_ind) for i in range(gates_num)]
    end_gate_inds = [start_gate_inds[j] + time_gate_2nd_ind  for j in range(gates_num)]

    # set last gate to the end sample
    end_gate_inds[-1] = N

    # initialise middle gate time and gain function arrays
    t_rms_values   = np.zeros(gates_num+2)
    amp_rms_values = np.zeros(gates_num+2)

    # loop over every gate
    ivalue = 1
    for istart, iend in zip(start_gate_inds, end_gate_inds):
        t_rms_values[ivalue]    = 0.5*(istart + iend)
        amp_rms_values[ivalue] = np.sqrt(np.mean(np.square(trace[istart:iend])))
        ivalue += 1

    # set side values for interpolation
    t_rms_values[-1] = N
    amp_rms_values[0] = amp_rms_values[1]
    amp_rms_values[-1] = amp_rms_values[-2]

    # linear interpolation for the rms amp function for every sample N
    rms_func = np.interp(range(N), t_rms_values, amp_rms_values )

    # calculate the gained trace
    gained_trace = trace*(np.sqrt(np.mean(np.square(trace)))/rms_func)


    return gained_trace


def inst_agc(trace, time, time_gate = 500e-3 ):
    """

    rms_agc: apply instantanous automatic gain control for a given trace.

     Usage:
         gained_trace = agc(data,time,agc_type, time_gate)

     Parameters
     -----------
     data: np.ndarray
            Input seismic trace
     time: np.ndarray
            Time array
     time_gate: float <class 'float'>
            Time gate used for agc in sec. typecal values between 200-500ms.

     Returns
     -------
     gained_trace: np.ndarray
        trace after applying instansous AGC

        INST_AGC is python function written by Musab Al Hasani based on the book of Oz Yilmaz (https://wiki.seg.org/wiki/Gain_applications)

    """
    # determine time sampling and num of samples
    dt = time[1]-time[0]
    N = len(trace)

    # determine the number of sample of a given gate
    end_samples = int(time_gate/dt)

    # calculate gates number not including the last end_samples
    gates_num = N - end_samples

    # initialise gates begining and end indices
    time_gate_1st_ind = 0
    time_gate_2nd_ind = int(time_gate/dt)

    # construct lists for indices of gates corners
    start_gate_inds = [i for i in range(gates_num)]
    end_gate_inds = [start_gate_inds[j] + time_gate_2nd_ind  for j in range(gates_num)]

    #initialise gain function
    amp_inst_values = np.zeros(N)

    # loop over ever sample to calculate gain function
    ivalue = 0
    for istart, iend in zip(start_gate_inds, end_gate_inds):
        amp_inst_values[ivalue] = np.mean(np.abs(trace[istart:iend]))
        ivalue += 1
    amp_inst_values[-end_samples:] = (amp_inst_values[ivalue-1])

    # calculate gained trace
    gained_trace = trace*(np.sqrt(np.mean(np.square(trace)))/amp_inst_values)

    return gained_trace
