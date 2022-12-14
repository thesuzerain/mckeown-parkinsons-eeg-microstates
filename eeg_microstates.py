#!/usr/bin/python
# -*- coding: utf-8 -*-
# Information theoretic analysis EEG microstates
# Ref.: von Wegner et al., NeuroImage 2017, 158:99-111
# FvW (12/2014)

import argparse, os, sys, time
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt, welch
from scipy.stats import chi2, chi2_contingency
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from statsmodels.stats.multitest import multipletests
# in case you have a compiled Cython implementation:
# from mutinf.mutinf_cy import mutinf_cy_c_wrapper, mutinf_cy

def read_xyz(filename):
    """Read EEG electrode locations in xyz format

    Args:
        filename: full path to the '.xyz' file
    Returns:
        locs: n_channels x 3 (numpy.array)
    """
    ch_names = []
    locs = []
    with open(filename, 'r') as f:
        l = f.readline()  # header line
        while l:
            l = f.readline().strip().split("\t")
            if (l != ['']):
                ch_names.append(l[0])
                locs.append([float(l[1]), float(l[2]), float(l[3])])
            else:
                l = None
    return ch_names, np.array(locs)


def read_edf(filename):
    """Basic EDF file format reader

    EDF specifications: http://www.edfplus.info/specs/edf.html

    Args:
        filename: full path to the '.edf' file
    Returns:
        chs: list of channel names
        fs: sampling frequency in [Hz]
        data: EEG data as numpy.array (samples x channels)
    """

    def readn(n):
        """read n bytes."""
        return np.fromfile(fp, sep='', dtype=np.int8, count=n)

    def bytestr(bytes, i):
        """convert byte array to string."""
        return np.array([bytes[k] for k in range(i*8, (i+1)*8)]).tostring()

    fp = open(filename, 'r')
    x = np.fromfile(fp, sep='', dtype=np.uint8, count=256).tostring()
    header = {}
    header['version'] = x[0:8]
    header['patientID'] = x[8:88]
    header['recordingID'] = x[88:168]
    header['startdate'] = x[168:176]
    header['starttime'] = x[176:184]
    header['length'] = int(x[184:192]) # header length (bytes)
    header['reserved'] = x[192:236]
    header['records'] = int(x[236:244]) # number of records
    header['duration'] = float(x[244:252]) # duration of each record [sec]
    header['channels'] = int(x[252:256]) # ns - number of signals
    n_ch = header['channels']  # number of EEG channels
    header['channelname'] = (readn(16*n_ch)).tostring()
    header['transducer'] = (readn(80*n_ch)).tostring().split()
    header['physdime'] = (readn(8*n_ch)).tostring().split()
    header['physmin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['physmin'].append(float(bytestr(b, i)))
    header['physmax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['physmax'].append(float(bytestr(b, i)))
    header['digimin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['digimin'].append(int(bytestr(b, i)))
    header['digimax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['digimax'].append(int(bytestr(b, i)))
    header['prefilt'] = (readn(80*n_ch)).tostring().split()
    header['samples_per_record'] = []
    b = readn(8*n_ch)
    for i in range(n_ch): header['samples_per_record'].append(float(bytestr(b, i)))
    nr = header['records']
    n_per_rec = int(header['samples_per_record'][0])
    n_total = int(nr*n_per_rec*n_ch)
    fp.seek(header['length'],os.SEEK_SET)  # header end = data start
    data = np.fromfile(fp, sep='', dtype=np.int16, count=n_total)  # count=-1
    fp.close()

    # re-order
    data = np.reshape(data,(n_per_rec,n_ch,nr),order='F')
    data = np.transpose(data,(0,2,1))
    data = np.reshape(data,(n_per_rec*nr,n_ch),order='F')

    # convert to physical dimensions
    for k in range(data.shape[1]):
        d_min = float(header['digimin'][k])
        d_max = float(header['digimax'][k])
        p_min = float(header['physmin'][k])
        p_max = float(header['physmax'][k])
        if ((d_max-d_min) > 0):
            data[:,k] = p_min+(data[:,k]-d_min)/(d_max-d_min)*(p_max-p_min)

    print(header)
    return header['channelname'].split(),\
           header['samples_per_record'][0]/header['duration'],\
           data


def bp_filter(data, f_lo, f_hi, fs):
    """Digital band pass filter (6-th order Butterworth)

    Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data
    """
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  # Nyquist frequency
    b_lo = f_lo/f_ny  # normalized frequency [0..1]
    b_hi = f_hi/f_ny  # normalized frequency [0..1]
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt


def plot_data(data, fs):
    """Plot the data

    Args:
        data: numpy.array
        fs: sampling frequency [Hz]
    """
    t = np.arange(len(data))/fs # time axis in seconds
    fig = plt.figure(1, figsize=(20,4))
    fig.patch.set_facecolor('white')
    plt.plot(t, data, '-k', linewidth=1)
    plt.xlabel("time [s]", fontsize=24)
    plt.ylabel("potential [$\mu$V]", fontsize=24)
    plt.tight_layout()
    plt.show()


def plot_psd(data, fs, n_seg=1024):
    """Plot the power spectral density (Welch's method)

    Args:
        data: numpy.array
        fs: sampling frequency [Hz]
        n_seg: samples per segment, default=1024
    """
    freqs, psd = welch(data, fs, nperseg=n_seg)
    fig = plt.figure(1, figsize=(16,8))
    fig.patch.set_facecolor('white')
    plt.semilogy(freqs, psd, 'k', linewidth=3)
    #plt.loglog(freqs, psd, 'k', linewidth=3)
    #plt.xlim(freqs.min(), freqs.max())
    #plt.ylim(psd.min(), psd.max())
    plt.title("Power spectral density (Welch's method  n={:d})".format(n_seg))
    plt.tight_layout()
    plt.show()


def topo(data, n_grid=64, channels=[], locs=[]):
    """Interpolate EEG topography onto a regularly spaced grid

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: integer, interpolate to n_grid x n_grid array, default=64
    Returns:
        data_interpol: cubic interpolation of EEG topography, n_grid x n_grid
                       contains nan values
    """
    # channels, locs = read_xyz('cap_theirs_soojin_intersection.xyz')
    # channels, locs = read_xyz('cap_soojin.xyz')
    n_channels = len(channels)
    #locs /= np.sqrt(np.sum(locs**2,axis=1))[:,np.newaxis]
    locs /= np.linalg.norm(locs, 2, axis=1, keepdims=True)
    c = findstr('Cz', channels)[0]
    # print 'center electrode for interpolation: ' + channels[c]
    #w = np.sqrt(np.sum((locs-locs[c])**2, axis=1))
    w = np.linalg.norm(locs-locs[c], 2, axis=1)
    #arclen = 2*np.arcsin(w/2)
    arclen = np.arcsin(w/2.*np.sqrt(4.-w*w))

    temp = list(map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1]))


    phi = np.angle(temp)
    #phi = np.angle(map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1]))
    X = arclen*np.real(np.exp(1j*phi))
    Y = arclen*np.imag(np.exp(1j*phi))
    r = max([max(X),max(Y)])
    Xi = np.linspace(-r,r,n_grid)
    Yi = np.linspace(-r,r,n_grid)

    # l0 = locs[:,0]-locs[c][0]
    # l1 = locs[:,1]-locs[c][1]

    # data_ip = np.zeros(shape=(n_grid,n_grid))
    # print(X)
    # for x,y in zip(X,Y):
    #     print(int(16*x+32))
    #     data_ip[int(16*x+32),int(16*y+32)] = 1

    data_ip = griddata((X, Y), data, (Xi[None,:], Yi[:,None]), method='linear')
    return data_ip


def eeg2map(data,channels=[], locs=[]):
    """Interpolate and normalize EEG topography, ignoring nan values

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: interger, interpolate to n_grid x n_grid array, default=64
    Returns:
        top_norm: normalized topography, n_grid x n_grid
    """
    n_grid = 64
    top = topo(data, n_grid, channels=channels, locs=locs)
    mn = np.nanmin(top)
    mx = np.nanmax(top)
    top_norm = (top-mn)/(mx-mn)
    return top_norm


def clustering(data, fs, chs, locs, mode, n_clusters, doplot=False, plotid = 0):
    """EEG microstate clustering algorithms.

    Args:
        data: numpy.array (n_t, n_ch)
        fs  : sampling frequency [Hz]
        chs : list of channel name strings
        locs: numpy.array (n_ch, 2) of electrode locations (x, y)
        mode: clustering algorithm
        n_clusters: number of clusters
        doplot: boolean
    Returns:
        maps: n_maps x n_channels NumPy array
        L: microstate sequence (integers)
        gfp_peaks: locations of local GFP maxima
        gev: global explained variance
    """
    #print("\t[+] Clustering algorithm: {:s}".format(mode))

    # --- number of EEG channels ---
    n_ch = data.shape[1]
    #print("\t[+] EEG channels: n_ch = {:d}".format(n_ch))

    # --- normalized data ---
    data_norm = data - data.mean(axis=1, keepdims=True)
    data_norm /= data_norm.std(axis=1, keepdims=True)

    # --- GFP peaks ---
    gfp = np.nanstd(data, axis=1)
    gfp2 = np.sum(gfp**2) # normalizing constant
    gfp_peaks = locmax(gfp)
    data_cluster = data[gfp_peaks,:]
    #data_cluster = data_cluster[:100,:]
    data_cluster_norm = data_cluster - data_cluster.mean(axis=1, keepdims=True)
    data_cluster_norm /= data_cluster_norm.std(axis=1, keepdims=True)
    print("\t[+] Data format for clustering [GFP peaks, channels]: {:d} x {:d}"\
         .format(data_cluster.shape[0], data_cluster.shape[1]))

    start = time.time()

    if (mode == "aahc"):
        print("\n\t[+] Clustering algorithm: AAHC.")
        maps = aahc(data, n_clusters, doplot=False)

    if (mode == "kmeans"):
        print("\n\t[+] Clustering algorithm: mod. K-MEANS.")
        maps = kmeans(data, n_maps=n_clusters, n_runs=10, doplot=False)

    if (mode == "kmedoids"):
        print("\n\t[+] Clustering algorithm: K-MEDOIDS.")
        C = np.corrcoef(data_cluster_norm)
        C = C**2 # ignore EEG polarity
        kmed_maps = kmedoids(S=C, K=n_clusters, nruns=10, maxits=500)
        maps = np.array([data_cluster[kmed_maps[k].__int__(),:] for k in range(n_clusters)])
        del C, kmed_maps

    if (mode == "pca"):
        print("\n\t[+] Clustering algorithm: PCA.")
        params = {"n_components": n_clusters, "copy": True, "whiten": True, "svd_solver": "auto"}
        pca = PCA(**params) # SKLEARN
        pca.fit(data_cluster_norm)
        maps = np.array([pca.components_[k,:] for k in range(n_clusters)])
        #print "PCA - explained variance: " + str(pca.explained_variance_ratio_)
        #print "PCA - total explained variance: " + str(np.sum(pca.explained_variance_ratio_))
        del pca, params

        # SKLEARN: TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
        #params = {"n_components": n_clusters, "algorithm": "randomized", "n_iter": 5, "random_state": None, "tol": 0.0}
        #svd = TruncatedSVD(**params)
        #svd.fit(data_cluster_norm)
        #maps = svd.components_
        #print "explained variance (%): "
        #print explained_variance_ratio_
        ##print "explained variance: "
        #print explained_variance_
        #del svd, params

    if (mode == "ica"):
        print("\n\t[+] Clustering algorithm: Fast-ICA.")
        #''' Fast-ICA: algorithm= parallel;deflation, fun=logcosh;exp;cube
        params = {"n_components": n_clusters, "algorithm": "parallel", "whiten": True, "fun": "exp", "max_iter": 200}
        ica = FastICA(**params) # SKLEARN
        S_ = ica.fit_transform(data_cluster_norm)  # reconstructed signals
        A_ = ica.mixing_  # estimated mixing matrix
        IC_ = ica.components_
        print("data: " + str(data_cluster_norm.shape))
        print("ICs: " + str(IC_.shape))
        print("mixing matrix: " + str(A_.shape))
        maps = np.array([ica.components_[k,:] for k in range(n_clusters)])
        del ica, params

    end = time.time()
    delta_t = end - start
    print("\t[+] Computation time: {:.2f}".format(delta_t))

    print("DATA NORM")
    print(maps[:,0]) ##############################

    # --- microstate sequence ---
    C = np.dot(data_norm, maps.T)/n_ch
    
    print("Their C")
    print(C[:10,:])

    L = np.argmax(C**2, axis=1)
    print("Their L")
    print(L[:10])

    del C

    # --- GEV ---
    maps_norm = maps - maps.mean(axis=1, keepdims=True)
    maps_norm /= maps_norm.std(axis=1, keepdims=True)

    # --- correlation data, maps ---
    C = np.dot(data_norm, maps_norm.T)/n_ch
    
    #print("\tC.shape: " + str(C.shape))
    #print("\tC.min: {:.2f}   Cmax: {:.2f}".format(C.min(), C.max()))

    # --- GEV_k & GEV ---
    gev = np.zeros(n_clusters)
    for k in range(n_clusters):
        r = L==k
        gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
    print("\n\t[+] Global explained variance GEV = {:.3f}".format(gev.sum()))
    #for k in range(n_clusters): print("\t\tGEV_{:d}: {:.3f}".format(k, gev[k]))

    if doplot:
        plt.ion()
        # matplotlib's perceptually uniform sequential colormaps:
        # magma, inferno, plasma, viridis
        cm = plt.cm.magma
        fig, axarr = plt.subplots(1, n_clusters, figsize=(20,5))
        fig.patch.set_facecolor('white')
        for imap in range(n_clusters):
            axarr[imap].imshow(eeg2map(maps[imap, :],channels=chs, locs=locs), cmap=cm, origin='lower')
            axarr[imap].set_xticks([])
            axarr[imap].set_xticklabels([])
            axarr[imap].set_yticks([])
            axarr[imap].set_yticklabels([])
        title = "Microstate maps ({:s})".format(mode.upper())
        axarr[0].set_title(title, fontsize=16, fontweight="bold")
        plt.show()

        # --- assign map labels manually ---
        print("\n\t\t Assign map labels by writing which of the above is 0, which is 1, etc ")
        order_str = input("\n\t\tType 'reset' or assign map labels (e.g. 0, 2, 1, 3): ")

        if order_str == "reset":
            # Bad cluster: we recurse into doing it again
            maps, L, gfp_peaks, gev = clustering(data, fs, chs, locs, mode, n_clusters, doplot)
        else:
            order_str = order_str.replace(",", "")
            order_str = order_str.replace(" ", "")
            if (len(order_str) != n_clusters):
                if (len(order_str)==0):
                    print("\t\tEmpty input string.")
                else:
                    print("\t\tParsed manual input: {:s}".format(", ".join(order_str)))
                    print("\t\tNumber of labels does not equal number of clusters.")
                print("\t\tContinue using the original assignment...\n")
            else:
                order = np.zeros(n_clusters, dtype=int)
                for i, s in enumerate(order_str):
                    order[i] = int(s)
                print("\t\tRe-ordered labels: {:s}".format(", ".join(order_str)))
                # re-order return variables
                print('Order.... 2?') #######################
                print(order) #######################
                print(maps[order,:][:,0])#######################
                maps = maps[order,:]
                temp_L = np.copy(L) #######################
                for i in range(len(L)):
                    L[i] = order[L[i]]
                gev = gev[order]
                # Figure

                print(L[:50])
                print(temp_L[:50])
                fig, axarr = plt.subplots(1, n_clusters, figsize=(20,5))
                fig.patch.set_facecolor('white')
                for imap in range(n_clusters):
                    axarr[imap].imshow(eeg2map(maps[imap, :],channels=chs, locs=locs), cmap=cm, origin='lower')
                    axarr[imap].set_xticks([])
                    axarr[imap].set_xticklabels([])
                    axarr[imap].set_yticks([])
                    axarr[imap].set_yticklabels([])
                title = "re-ordered microstate maps"
                axarr[0].set_title(title, fontsize=16, fontweight="bold")
                plt.savefig('sub/sub'+str(plotid)+'.png')
                plt.show()
                plt.ioff()

    return maps, L, gfp_peaks, gev


def aahc(data, N_clusters, doplot=False):
    """Atomize and Agglomerative Hierarchical Clustering Algorithm
    AAHC (Murray et al., Brain Topography, 2008)

    Args:
        data: EEG data to cluster, numpy.array (n_samples, n_channels)
        N_clusters: desired number of clusters
        doplot: boolean, plot maps
    Returns:
        maps: n_maps x n_channels (numpy.array)
    """

    def extract_row(A, k):
        v = A[k,:]
        A_ = np.vstack((A[:k,:],A[k+1:,:]))
        return A_, v

    def extract_item(A, k):
        a = A[k]
        A_ = A[:k] + A[k+1:]
        return A_, a

    #print("\n\t--- AAHC ---")
    nt, nch = data.shape

    # --- get GFP peaks ---
    gfp = data.std(axis=1)
    gfp_peaks = locmax(gfp)
    #gfp_peaks = gfp_peaks[:100]
    #n_gfp = gfp_peaks.shape[0]
    gfp2 = np.sum(gfp**2) # normalizing constant in GEV

    # --- initialize clusters ---
    maps = data[gfp_peaks,:]
    # --- store original gfp peaks and indices ---
    cluster_data = data[gfp_peaks,:]
    #n_maps = n_gfp
    n_maps = maps.shape[0]
    print("\t[+] Initial number of clusters: {:d}\n".format(n_maps))

    # --- cluster indices w.r.t. original size, normalized GFP peak data ---
    Ci = [[k] for k in range(n_maps)]

    # --- main loop: atomize + agglomerate ---
    while (n_maps > N_clusters):
        s = "\r{:s}\r\t\tAAHC > n: {:d} => {:d}".format(80*" ", n_maps, n_maps-1)
        stdout.write(s); stdout.flush()
        #print("\n\tAAHC > n: {:d} => {:d}".format(n_maps, n_maps-1))

        # --- correlations of the data sequence with each cluster ---
        m_x, s_x = data.mean(axis=1, keepdims=True), data.std(axis=1)
        m_y, s_y = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
        s_xy = 1.*nch*np.outer(s_x, s_y)
        C = np.dot(data-m_x, np.transpose(maps-m_y)) / s_xy

        # --- microstate sequence, ignore polarity ---
        L = np.argmax(C**2, axis=1)

        # --- GEV (global explained variance) of cluster k ---
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2

        # --- merge cluster with the minimum GEV ---
        imin = np.argmin(gev)
        #print("\tre-cluster: {:d}".format(imin))

        # --- N => N-1 ---
        maps, _ = extract_row(maps, imin)
        Ci, reC = extract_item(Ci, imin)
        re_cluster = []  # indices of updated clusters
        #C_sgn = np.zeros(nt)
        for k in reC:  # map index to re-assign
            c = cluster_data[k,:]
            m_x, s_x = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
            m_y, s_y = c.mean(), c.std()
            s_xy = 1.*nch*s_x*s_y
            C = np.dot(maps-m_x, c-m_y)/s_xy
            inew = np.argmax(C**2) # ignore polarity
            #C_sgn[k] = C[inew]
            re_cluster.append(inew)
            Ci[inew].append(k)
        n_maps = len(Ci)

        # --- update clusters ---
        re_cluster = list(set(re_cluster)) # unique list of updated clusters

        ''' re-clustering by modified mean
        for i in re_cluster:
            idx = Ci[i]
            c = np.zeros(nch) # new cluster average
            for k in idx: # add to new cluster, polarity according to corr. sign
                if (C_sgn[k] >= 0):
                    c += cluster_data[k,:]
                else:
                    c -= cluster_data[k,:]
            c /= len(idx)
            maps[i] = c
            #maps[i] = (c-np.mean(c))/np.std(c) # normalize the new cluster
        del C_sgn
        '''

        # re-clustering by eigenvector method
        for i in re_cluster:
            idx = Ci[i]
            Vt = cluster_data[idx,:]
            Sk = np.dot(Vt.T, Vt)
            evals, evecs = np.linalg.eig(Sk)
            c = evecs[:, np.argmax(np.abs(evals))]
            c = np.real(c)
            maps[i] = c/np.sqrt(np.sum(c**2))

    print()
    return maps


def kmeans(data, n_maps, n_runs=10, maxerr=1e-6, maxiter=500, doplot=False):
    """Modified K-means clustering as detailed in:
    [1] Pascual-Marqui et al., IEEE TBME (1995) 42(7):658--665
    [2] Murray et al., Brain Topography(2008) 20:249--264.
    Variables named as in [1], step numbering as in Table I.

    Args:
        data: numpy.array, size = number of EEG channels
        n_maps: number of microstate maps
        n_runs: number of K-means runs (optional)
        maxerr: maximum error for convergence (optional)
        maxiter: maximum number of iterations (optional)
        doplot: plot the results, default=False (optional)
    Returns:
        maps: microstate maps (number of maps x number of channels)
        L: sequence of microstate labels
        gfp_peaks: indices of local GFP maxima
        gev: global explained variance (0..1)
        cv: value of the cross-validation criterion
    """
    n_t = data.shape[0]
    n_ch = data.shape[1]
    data = data - data.mean(axis=1, keepdims=True)

    # GFP peaks
    gfp = np.std(data, axis=1)
    gfp_peaks = locmax(gfp)
    gfp_values = gfp[gfp_peaks]
    gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
    n_gfp = gfp_peaks.shape[0]

    # clustering of GFP peak maps only
    V = data[gfp_peaks, :]
    sumV2 = np.sum(V**2)

    # store results for each k-means run
    cv_list =   []  # cross-validation criterion for each k-means run
    gev_list =  []  # GEV of each map for each k-means run
    gevT_list = []  # total GEV values for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list =    []  # microstate label sequence for each k-means run
    for run in range(n_runs):
        # initialize random cluster centroids (indices w.r.t. n_gfp)
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ( (np.abs((var0-var1)/var0) > maxerr) & (n_iter < maxiter) ):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_ch*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
            L = np.argmax(C**2, axis=1)
            # (step 4)
            for k in range(n_maps):
                Vt = V[L==k, :]
                # (step 4a)
                Sk = np.dot(Vt.T, Vt)
                # (step 4b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                maps[k, :] = v/np.sqrt(np.sum(v**2))
            # (step 5)
            var1 = var0
            var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
            var0 /= (n_gfp*(n_ch-1))
            n_iter += 1
        if (n_iter < maxiter):
            print("\t\tK-means run {:d}/{:d} converged after {:d} iterations.".format(run+1, n_runs, n_iter))
        else:
            print("\t\tK-means run {:d}/{:d} did NOT converge after {:d} iterations.".format(run+1, n_runs, maxiter))

        # CROSS-VALIDATION criterion for this run (step 8)
        C_ = np.dot(data, maps.T)
        C_ /= (n_ch*np.outer(gfp, np.std(maps, axis=1)))
        L_ = np.argmax(C_**2, axis=1)
        var = np.sum(data**2) - np.sum(np.sum(maps[L_, :]*data, axis=1)**2)
        var /= (n_t*(n_ch-1))
        cv = var * (n_ch-1)**2/(n_ch-n_maps-1.)**2

        # GEV (global explained variance) of cluster k
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp_values[r]**2 * C[r,k]**2)/gfp2
        gev_total = np.sum(gev)

        # store
        cv_list.append(cv)
        gev_list.append(gev)
        gevT_list.append(gev_total)
        maps_list.append(maps)
        L_list.append(L_)

    # select best run
    k_opt = np.argmin(cv_list)
    #k_opt = np.argmax(gevT_list)
    maps = maps_list[k_opt]
    # ms_gfp = ms_list[k_opt] # microstate sequence at GFP peaks
    gev = gev_list[k_opt]
    L_ = L_list[k_opt]

    if doplot:
        plt.ion()
        # matplotlib's perceptually uniform sequential colormaps:
        # magma, inferno, plasma, viridis
        cm = plt.cm.magma
        fig, axarr = plt.subplots(1, n_maps, figsize=(20,5))
        fig.patch.set_facecolor('white')
        for imap in range(n_maps):
            axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
            axarr[imap].set_xticks([])
            axarr[imap].set_xticklabels([])
            axarr[imap].set_yticks([])
            axarr[imap].set_yticklabels([])
        title = "K-means cluster centroids"
        axarr[0].set_title(title, fontsize=16, fontweight="bold")
        plt.show()

        # --- assign map labels manually ---
        order_str = raw_input("\n\t\tAssign map labels (e.g. 0, 2, 1, 3): ")
        order_str = order_str.replace(",", "")
        order_str = order_str.replace(" ", "")
        if (len(order_str) != n_maps):
            if (len(order_str)==0):
                print("\t\tEmpty input string.")
            else:
                print("\t\tParsed manual input: {:s}".format(", ".join(order_str)))
                print("\t\tNumber of labels does not equal number of clusters.")
            print("\t\tContinue using the original assignment...\n")
        else:
            order = np.zeros(n_maps, dtype=int)
            for i, s in enumerate(order_str):
                order[i] = int(s)
            print("\t\tRe-ordered labels: {:s}".format(", ".join(order_str)))
            # re-order return variables
            maps = maps[order,:]
            for i in range(len(L)):
                L[i] = order[L[i]]
            gev = gev[order]
            # Figure
            print('Order....?')
            fig, axarr = plt.subplots(1, n_maps, figsize=(20,5))
            fig.patch.set_facecolor('white')
            for imap in range(n_maps):
                axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
                axarr[imap].set_xticks([])
                axarr[imap].set_xticklabels([])
                axarr[imap].set_yticks([])
                axarr[imap].set_yticklabels([])
            title = "re-ordered K-means cluster centroids"
            axarr[0].set_title(title, fontsize=16, fontweight="bold")
            plt.show()
            plt.ioff()
    #return maps, L_, gfp_peaks, gev, cv
    return maps


def kmedoids(S, K, nruns, maxits):
    """Octave/Matlab: Copyright Brendan J. Frey and Delbert Dueck, Aug 2006
    http://www.psi.toronto.edu/~frey/apm/kcc.m
    Simplified by Kevin Murphy
    Python 2.7.x - FvW, 02/2014

    Args:
        filename: full path to the '.xyz' file
    Returns:
        locs: n_channels x 3 (numpy.array)
    """
    n = S.shape[0]
    dpsim = np.zeros((maxits,nruns))
    idx = np.zeros((n,nruns))
    for rep in xrange(nruns):
        tmp = np.random.permutation(range(n))
        mu = tmp[:K]
        t = 0
        done = (t==maxits)
        while ( not done ):
            t += 1
            muold = mu
            dpsim[t,rep] = 0
            # Find class assignments
            cl = np.argmax(S[:,mu], axis=1) # max pos. of each row
            # Set assignments of exemplars to themselves
            cl[mu] = range(K)
            for j in xrange(K): # For each class, find new exemplar
                I = np.where(cl==j)[0]
                S_I_rowsum = np.sum(S[I][:,I],axis=0)
                Scl = max(S_I_rowsum)
                ii = np.argmax(S_I_rowsum)
                dpsim[t,rep] = dpsim[t,rep] + Scl
                mu[j] = I[ii]
            if all(muold==mu) | (t==maxits):
                done = 1
        idx[:,rep] = mu[cl]
        dpsim[t+1:,rep] = dpsim[t,rep]
    return np.unique(idx)


def p_empirical(data, n_clusters):
    """Empirical symbol distribution

    Args:
        data: numpy.array, size = length of microstate sequence
        n_clusters: number of microstate clusters
    Returns:
        p: empirical distribution
    """
    p = np.zeros(n_clusters)
    n = len(data)
    for i in range(n):
        p[data[i]] += 1.0
    p /= n
    return p


def T_empirical(data, n_clusters):
    """Empirical transition matrix

    Args:
        data: numpy.array, size = length of microstate sequence
        n_clusters: number of microstate clusters
    Returns:
        T: empirical transition matrix
    """
    T = np.zeros((n_clusters, n_clusters))
    n = len(data)
    for i in range(n-1):
        T[data[i], data[i+1]] += 1.0
    p_row = np.sum(T, axis=1)
    for i in range(n_clusters):
        if ( p_row[i] != 0.0 ):
            for j in range(n_clusters):
                T[i,j] /= p_row[i]  # normalize row sums to 1.0
    return T


def p_equilibrium(T):
    '''get equilibrium distribution from transition matrix: lambda=1 - (left) eigenvector'''
    evals, evecs = np.linalg.eig(T.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0] # locate max eigenval.
    p_eq = np.abs(evecs[:,i]) # make eigenvec. to max. eigenval. non-negative
    p_eq /= p_eq.sum() # normalized eigenvec. to max. eigenval.
    return p_eq # stationary distribution


def max_entropy(n_clusters):
    """Maximum Shannon entropy of a sequence with n_clusters

    Args:
        n_clusters: number of microstate clusters
    Returns:
        h_max: maximum Shannon entropy
    """
    h_max = np.log(float(n_clusters))
    return h_max


def shuffle(data):
    """Randomly shuffled copy of data (i.i.d. surrogate)

    Args:
        data: numpy array, 1D
    Returns:
        data_copy: shuffled data copy
    """
    data_c = data.copy()
    np.random.shuffle(data_c)
    return data_c


def surrogate_mc(p, T, n_clusters, n):
    """Surrogate Markov chain with symbol distribution p and
    transition matrix T

    Args:
        p: empirical symbol distribution
        T: empirical transition matrix
        n_clusters: number of clusters/symbols
        n: length of surrogate microstate sequence
    Returns:
        mc: surrogate Markov chain
    """

    # NumPy vectorized code
    psum = np.cumsum(p)
    Tsum = np.cumsum(T, axis=1)
    # initial state according to p:
    mc = [np.min(np.argwhere(np.random.rand() < psum))]
    # next state according to T:
    for i in range(1, n):
        mc.append(np.min(np.argwhere(np.random.rand() < Tsum[mc[i-1]])))

    ''' alternative implementation using loops
    r = np.random.rand() # ~U[0,1], random threshold
    s = 0.
    y = p[s]
    while (y < r):
        s += 1.
        y += p[s]
    mc = [s] # initial state according to p

    # iterate ...
    for i in xrange(1,n):
        r = np.random.rand() # ~U[0,1], random threshold
        s = mc[i-1] # currrent state
        t = 0. # trial state
        y = T[s][t] # transition rate to trial state
        while ( y < r ):
            t += 1. # next trial state
            y += T[s][t]
        mc.append(t) # store current state
    '''

    return np.array(mc)


def mutinf(x, ns, lmax):
    """Time-lagged mutual information of symbolic sequence x with
    ns different symbols, up to maximum lag lmax.
    *** Symbols must be 0, 1, 2, ... to use as indices directly! ***

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        lmax: maximum time lag
    Returns:
        mi: time-lagged mutual information
    """

    n = len(x)
    mi = np.zeros(lmax)
    for l in range(lmax):
        if ((l+1)%10 == 0):
            s = "\r\t\tmutual information lag: {:d}".format(l+1)
            sys.stdout.write(s)
            sys.stdout.flush()
        nmax = n-l
        h1 = H_1(x[:nmax], ns)
        h2 = H_1(x[l:l+nmax], ns)
        h12 = H_2(x[:nmax], x[l:l+nmax], ns)
        mi[l] = h1 + h2 - h12
    return mi


def mutinf_i(x, ns, lmax):
    """Time-lagged mutual information for each symbol of the symbolic
    sequence x with  ns different symbols, up to maximum lag lmax.
    *** Symbols must be 0, 1, 2, ... to use as indices directly! ***

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        lmax: maximum time lag
    Returns:
        mi: time-lagged mutual informations for each symbol, ns x lmax
    """

    n = len(x)
    mi = np.zeros((ns,lmax))
    for l in range(lmax):
        if (l%10 == 0):
            s = "\r\t\tmutual information lag: {:d}".format(l)
            sys.stdout.write(s)
            sys.stdout.flush()
        nmax = n - l

        # compute distributions
        p1 = np.zeros(ns)
        p2 = np.zeros(ns)
        p12 = np.zeros((ns,ns))
        for i in range(nmax):
            i1 = int(x[i])
            i2 = int(x[i+l])
            p1[i1] += 1.0 # p( X_t = i1 )
            p2[i2] += 1.0 # p( X_t+l = i2 )
            p12[i1,i2] += 1.0 # p( X_t+l=i2 , X_t=i1 )
        p1 /= nmax
        p2 /= nmax

        # normalize the transition matrix p( X_t+l=i2 | X_t=i1 )
        rowsum = np.sum(p12, axis=1)
        for i, j in np.ndindex(p12.shape):
            p12[i,j] /= rowsum[i]

        # compute entropies
        H2 = np.sum(p2[p2>0] * np.log(p2[p2>0]))
        for i in range(ns):
            H12 = 0.0
            for j in range(ns):
                if ( p12[i,j] > 0.0 ):
                    H12 += ( p12[i,j] * np.log( p12[i,j] ) )
            mi[i,l] = -H2 + p1[i] * H12
    return mi


def mutinf_CI(p, T, n, alpha, nrep, lmax):
    """Return an array for the computation of confidence intervals (CI) for
    the time-lagged mutual information of length lmax.
    Null Hypothesis: Markov Chain of length n, equilibrium distribution p,
    transition matrix T, using nrep repetitions."""

    ns = len(p)
    mi_array = np.zeros((nrep,lmax))
    for r in range(nrep):
        print("\n\t\tsurrogate MC # {:d} / {:d}".format(r+1, nrep))
        x_mc = surrogate_mc(p, T, ns, n)
        mi_mc = mutinf(x_mc, ns, lmax)
        #mi_mc = mutinf_cy(X_mc, ns, lmax)
        mi_array[r,:] = mi_mc
    return mi_array


def excess_entropy_rate(x, ns, kmax, doplot=False):
    # y = ax+b: line fit to joint entropy for range of histories k
    # a = entropy rate (slope)
    # b = excess entropy (intersect.)
    h_ = np.zeros(kmax)
    for k in range(kmax): h_[k] = H_k(x, ns, k+1)
    ks = np.arange(1,kmax+1)
    a, b = np.polyfit(ks, h_, 1)
    # --- Figure ---
    if doplot:
        plt.figure(figsize=(6,6))
        plt.plot(ks, h_, '-sk')
        plt.plot(ks, a*ks+b, '-b')
        plt.xlabel("k")
        plt.ylabel("$H_k$")
        plt.title("Entropy rate")
        plt.tight_layout()
        plt.show()
    return (a, b)


def mc_entropy_rate(p, T):
    """Markov chain entropy rate.
    - \sum_i sum_j p_i T_ij log(T_ij)
    """
    h = 0.
    for i, j in np.ndindex(T.shape):
        if (T[i,j] > 0):
            h -= ( p[i]*T[i,j]*np.log(T[i,j]) )
    return h


def aif_peak1(mi, fs, doplot=False):
    '''compute time-lagged mut. inf. (AIF) and 1st peak.'''
    dt = 1000./fs # sampling interval [ms]
    mi_filt = np.convolve(mi, np.ones(3)/3., mode='same')
    #mi_filt = np.convolve(mi, np.ones(5)/5., mode='same')
    mx0 = 8 # 8
    jmax = mx0 + locmax(mi_filt[mx0:])[0]
    mx_mi = dt*jmax
    if doplot:
        offset = 5
        tmax = 100
        fig = plt.figure(1, figsize=(22,4))
        fig.patch.set_facecolor('white')
        t = dt*np.arange(tmax)
        plt.plot(t[offset:tmax], mi[offset:tmax], '-ok', label='AIF')
        plt.plot(t[offset:tmax], mi_filt[offset:tmax], '-b', label='smoothed AIF')
        plt.plot(mx_mi, mi[jmax], 'or', markersize=15, label='peak-1')
        plt.xlabel("time lag [ms]")
        plt.ylabel("mut. inf. [bits]")
        plt.legend(loc=0)
        #plt.title("mutual information of map sequence")
        #plt.title(s, fontsize=16, fontweight='bold')
        plt.show()
        raw_input()

    return jmax, mx_mi


def H_1(x, ns):
    """Shannon entropy of the symbolic sequence x with ns symbols.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """

    n = len(x)
    p = np.zeros(ns) # symbol distribution
    for t in range(n):
        p[x[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*np.log(p[p>0]))
    return h


def H_2(x, y, ns):
    """Joint Shannon entropy of the symbolic sequences X, Y with ns symbols.

    Args:
        x, y: symbolic sequences, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """

    if (len(x) != len(y)):
        print("H_2 warning : sequences of different lengths, using the shorter...")
    n = min([len(x), len(y)])
    p = np.zeros((ns, ns)) # joint distribution
    for t in range(n):
        p[x[t],y[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*np.log(p[p>0]))
    return h


def H_k(x, ns, k):
    """Shannon's joint entropy from x[n+p:n-m]
    x: symbolic time series
    ns: number of symbols
    k: length of k-history
    """

    N = len(x)
    f = np.zeros(tuple(k*[ns]))
    for t in range(N-k): f[tuple(x[t:t+k])] += 1.0
    f /= (N-k) # normalize distribution
    hk = -np.sum(f[f>0]*np.log(f[f>0]))
    #m = np.sum(f>0)
    #hk = hk + (m-1)/(2*N) # Miller-Madow bias correction
    return hk


def testMarkov0(x, ns, alpha, verbose=True):
    """Test zero-order Markovianity of symbolic sequence x with ns symbols.
    Null hypothesis: zero-order MC (iid) <=>
    p(X[t]), p(X[t+1]) independent
    cf. Kullback, Technometrics (1962)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\t\tZERO-ORDER MARKOVIANITY:" )
    n = len(x)
    f_ij = np.zeros((ns,ns))
    f_i = np.zeros(ns)
    f_j = np.zeros(ns)
    # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
    for t in range(n-1):
        i = x[t]
        j = x[t+1]
        f_ij[i,j] += 1.0
        f_i[i] += 1.0
        f_j[j] += 1.0
    T = 0.0 # statistic
    for i, j in np.ndindex(f_ij.shape):
        f = f_ij[i,j]*f_i[i]*f_j[j]
        if (f > 0):
            T += (f_ij[i,j] * np.log((n*f_ij[i,j])/(f_i[i]*f_j[j])))
    T *= 2.0
    df = (ns-1.0) * (ns-1.0)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df))
    return p


def testMarkov1(X, ns, alpha, verbose=True):
    """Test first-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\n\t\tFIRST-ORDER MARKOVIANITY:" )
    n = len(X)
    f_ijk = np.zeros((ns,ns,ns))
    f_ij = np.zeros((ns,ns))
    f_jk = np.zeros((ns,ns))
    f_j = np.zeros(ns)
    for t in range(n-2):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        f_ijk[i,j,k] += 1.0
        f_ij[i,j] += 1.0
        f_jk[j,k] += 1.0
        f_j[j] += 1.0
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i][j][k]*f_j[j]*f_ij[i][j]*f_jk[j][k]
        if (f > 0):
            T += (f_ijk[i,j,k]*np.log((f_ijk[i,j,k]*f_j[j])/(f_ij[i,j]*f_jk[j,k])))
    T *= 2.0
    df = ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df))
    return p


def testMarkov2(X, ns, alpha, verbose=True):
    """Test second-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t], X[t-1]) = p(X[t+1] | X[t], X[t-1], X[t-2])
    cf. Kullback, Technometrics (1962), Table 10.2.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\n\t\tSECOND-ORDER MARKOVIANITY:" )
    n = len(X)
    f_ijkl = np.zeros((ns,ns,ns,ns))
    f_ijk = np.zeros((ns,ns,ns))
    f_jkl = np.zeros((ns,ns,ns))
    f_jk = np.zeros((ns,ns))
    for t in range(n-3):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        l = X[t+3]
        f_ijkl[i,j,k,l] += 1.0
        f_ijk[i,j,k] += 1.0
        f_jkl[j,k,l] += 1.0
        f_jk[j,k] += 1.0
    T = 0.0
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        f = f_ijkl[i,j,k,l]*f_ijk[i,j,k]*f_jkl[j,k,l]*f_jk[j,k]
        if (f > 0):
            T += (f_ijkl[i,j,k,l]*np.log((f_ijkl[i,j,k,l]*f_jk[j,k])/(f_ijk[i,j,k]*f_jkl[j,k,l])))
    T *= 2.0
    df = ns*ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df))
    return p


def conditionalHomogeneityTest(X, ns, l, alpha, verbose=True):
    """Test conditional homogeneity of non-overlapping blocks of
    length l of symbolic sequence X with ns symbols
    cf. Kullback, Technometrics (1962), Table 9.1.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        l: split x into non-overlapping blocks of size l
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print("\n\t\tCONDITIONAL HOMOGENEITY (three-way table):")
    n = len(X)
    r = int(np.floor(float(n)/float(l))) # number of blocks
    nl = r*l
    if verbose:
        print("\t\tSplit data in r = {:d} blocks of length {:d}.".format(r,l))
    f_ijk = np.zeros((r,ns,ns))
    f_ij = np.zeros((r,ns))
    f_jk = np.zeros((ns,ns))
    f_i = np.zeros(r)
    f_j = np.zeros(ns)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in  range(r): # block index
        for ii in range(l-1): # pos. inside the current block
            j = X[i*l + ii]
            k = X[i*l + ii + 1]
            f_ijk[i,j,k] += 1.0
            f_ij[i,j] += 1.0
            f_jk[j,k] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0

    # conditional homogeneity (Markovianity stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i,j,k]*f_j[j]*f_ij[i,j]*f_jk[j,k]
        if (f > 0):
            T += (f_ijk[i,j,k]*np.log((f_ijk[i,j,k]*f_j[j])/(f_ij[i,j]*f_jk[j,k])))
    T *= 2.0
    df = (r-1)*(ns-1)*ns
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df))
    return p


def symmetryTest(X, ns, alpha, verbose=True):
    """Test symmetry of the transition matrix of symbolic sequence X with
    ns symbols
    cf. Kullback, Technometrics (1962)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print( "\n\t\tSYMMETRY:" )
    n = len(X)
    f_ij = np.zeros((ns,ns))
    for t in range(n-1):
        i = X[t]
        j = X[t+1]
        f_ij[i,j] += 1.0
    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i,j]*f_ij[j,i]
            if (f > 0):
                T += (f_ij[i,j]*np.log((2.*f_ij[i,j])/(f_ij[i,j]+f_ij[j,i])))
    T *= 2.0
    df = ns*(ns-1)/2
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p, T, df))
    return p


def geoTest(X, ns, dt, alpha, verbose=True):
    """Test the geometric distribution of lifetime distributions.

    Args:
        X: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        dt: time step in [ms]
        alpha: significance level
    Returns:
        p_values: for each symbol
    """

    if verbose:
        print("\n\t\tGEOMETRIC DISTRIBUTION of lifetimes:\n")
    tauDist = lifetimes(X, ns)
    T = T_empirical(X, ns)
    p_values = np.zeros(ns)
    for s in range(ns): # test for each symbol
        if verbose:
            print( "\n\t\tTesting the distribution of symbol # {:d}".format(s) )
        # m = max_tau:
        m = len(tauDist[s])
        # observed lifetime distribution:
        q_obs = np.zeros(m)
        # theoretical lifetime distribution:
        q_exp = np.zeros(m)
        for j in range(m):
            # observed frequency of lifetime j+1 for state s
            q_obs[j] = tauDist[s][j]
            # expected frequency
            q_exp[j] = (1.0-T[s][s]) * (T[s][s]**j)
        q_exp *= sum(q_obs)

        t = 0.0 # chi2 statistic
        for j in range(m):
            if ((q_obs[j] > 0) & (q_exp[j] > 0)):
                t += (q_obs[j]*np.log(q_obs[j]/q_exp[j]))
        t *= 2.0
        df = m-1
        #p0 = chi2test(t, df, alpha)
        p0 = chi2.sf(t, df, loc=0, scale=1)
        if verbose:
            print("\t\tp: {:.2e} | t: {:.3f} | df: {:.1f}".format(p0,t,df))
        p_values[s] = p0

        g1, p1, dof1, expctd1 = chi2_contingency( np.vstack((q_obs,q_exp)), lambda_='log-likelihood' )
        if verbose:
            print("\t\tG-test (log-likelihood) p: {:.2e}, g: {:.3f}, df: {:.1f}".format(p1,g1,dof1))

        g2, p2, dof2, expctd2 = chi2_contingency( np.vstack((q_obs,q_exp)) ) # Pearson's Chi2 test
        if verbose:
            print("\t\tG-test (Pearson Chi2) p: {:.2e}, g: {:.3f}, df: {:.1f}".format(p2,g2,dof2))

        p_ = tauDist[s]/np.sum(tauDist[s])
        tau_mean = 0.0
        tau_max = len(tauDist[s])*dt
        for j in range(len(p_)): tau_mean += (p_[j] * j)
        tau_mean *= dt
        if verbose:
            pass
            #print("\t\tmean dwell time: {:.2f} [ms]".format(tau_mean))
        if verbose:
            pass
            #print("\t\tmax. dwell time: {:.2f} [ms]\n\n".format(tau_max))
    return p_values


def lifetimes(X, ns):
    """Compute the lifetime distributions for each symbol
    in a symbolic sequence X with ns symbols.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        tauDist: list of lists, lifetime distributions for each symbol
    """

    n = len(X)
    tauList = [[] for i in range(ns)] # unsorted lifetimes, [[],[],[],[]]
    i = 0 # current index
    s = X[i] # current symbol
    tau = 1.0 # current lifetime
    while (i < n-1):
        i += 1
        if ( X[i] == s ):
            tau += 1.0
        else:
            tauList[int(s)].append(tau)
            s = X[i]
            tau = 1.0
    tauList[int(s)].append(tau) # last state
    # find max lifetime for each symbol
    tau_max = [max(L) for L in tauList]
    #print( "max lifetime for each symbol : " + str(tau_max) )
    tauDist = [] # empty histograms for each symbol
    for i in range(ns): tauDist.append( [0.]*int(tau_max[i]) )
    # lifetime distributions
    for s in range(ns):
        for j in range(len(tauList[s])):
            tau = tauList[s][j]
            tauDist[s][int(tau)-1] += 1.0
    return tauDist


def multiple_comparisons(p_values, method):
    """Apply multiple comparisons correction code using the statsmodels package.
    Input: array p-values.
    'b': 'Bonferroni'
    's': 'Sidak'
    'h': 'Holm'
    'hs': 'Holm-Sidak'
    'sh': 'Simes-Hochberg'
    'ho': 'Hommel'
    'fdr_bh': 'FDR Benjamini-Hochberg'
    'fdr_by': 'FDR Benjamini-Yekutieli'
    'fdr_tsbh': 'FDR 2-stage Benjamini-Hochberg'
    'fdr_tsbky': 'FDR 2-stage Benjamini-Krieger-Yekutieli'
    'fdr_gbs': 'FDR adaptive Gavrilov-Benjamini-Sarkar'

    Args:
        p_values: uncorrected p-values
        method: string, one of the methods given above
    Returns:
        p_values_corrected: corrected p-values
    """
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, method=method)
    return pvals_corrected


def locmax(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """

    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m


def findstr(s, L):
    """Find string in list of strings, returns indices.

    Args:
        s: query string
        L: list of strings to search
    Returns:
        x: list of indices where s is found in L
    """

    x = [i for i, l in enumerate(L) if (l==s)]
    return x


def print_matrix(T):
    """Console-friendly output of the matrix T.

    Args:
        T: matrix to print
    """

    for i, j in np.ndindex(T.shape):
        if (j == 0):
            print ("\t\t|{:.3f}").format(T[i,j]),
        elif (j == T.shape[1]-1):
            print ("{:.3f}|\n").format(T[i,j]),
        else:
            print ("{:.3f}").format(T[i,j]),


def main():
    """Test module functions."""

    os.system("clear")
    print("\n\n\t\t--- EEG microstate analysis ---\n\n")
    parser = argparse.ArgumentParser(description='EEG microstate analysis')
    parser.add_argument('-i','--input', help='.edf file name', required=False)
    parser.add_argument('-f','--filelist', help='.edf file list', required=False)
    parser.add_argument('-d', '--directory', \
                        help='process all .edf files in this directory', \
                        required=False)
    parser.add_argument('-m', '--markovsurrogates', \
                        help='number of Markov surrogates', required=False)
    args = parser.parse_args()
    argsdict = vars(args)
    print("\n\tTutorial: press <Enter> to proceed to the next step\n")

    # (0) make file list
    if argsdict['input']:
        filenames = [argsdict['input']]
    elif argsdict['filelist']:
        with open(argsdict['filelist'], 'rb') as f:
            filenames = [l.strip() for l in f]
    elif argsdict['directory']:
        filenames = []
        for f1 in os.listdir(argsdict['directory']):
            f2 = os.path.join(argsdict['directory'], f1)
            if os.path.isfile(f2):
                if (f2[-4:] == ".edf"):
                    filenames.append(f2)
    else:
        filenames = ["test.edf"]

    print("\tList of file names generated:")
    print("\t" + str(filenames))
    raw_input("\n\t\tpress ENTER to continue...")
    # define lists that hold p-values for multiple comparisons
    p_Markov0 = []
    p_Markov1 = []
    p_Markov2 = []
    p_geo = []
    p_stationarity = []
    p_symmetry = []

    # process files
    for filename in filenames:

        # (1) load edf file
        os.system("clear")
        print("\n\t[1] Load EEG file:")
        chs, fs, data_raw = read_edf(filename)
        print("\n\tFile: {}".format(filename))
        print("\tSampling rate: {:.2f} Hz".format(fs))
        print("\tData shape: {:d} samples x {:d} channels".\
        format(data_raw.shape[0], data_raw.shape[1]))
        print("\tChannels:")
        for ch in chs:
            print ("\t{:s}").format(ch),
        raw_input("\n\n\n\t\tpress ENTER to continue...")

        # (2) band-pass filter 1-35 Hz (6-th order Butterworth)
        os.system("clear")
        print("\n\t[2] Apply band-pass filter...")
        data = bp_filter(data_raw, (1, 35), fs) # (1, 35)
        #data = np.copy(data_raw)
        raw_input("\n\n\t\tpress ENTER to continue...")

        #''' (3) visualize data
        os.system("clear")
        print("\n\t[3] Visualize data (PCA-1):")
        pca = PCA(copy=True, n_components=1, whiten=False)
        pca1 = pca.fit_transform(data)[:,0]
        del pca
        plot_data(pca1, fs)
        #raw_input("\n\n\t\tpress ENTER to continue...")
        #'''

        # (4) microstate clustering
        os.system("clear")
        print("\n\t[4] Clustering EEG topographies at GFP peaks:")
        n_maps = 4
        #maps, x, gfp_peaks, gev, cv = kmeans(data, n_maps=n_maps, n_runs=5, \
        #                                      doplot=True)
        locs = []

        # choose clustering method
        mode = ["aahc", "kmeans", "kmedoids", "pca", "ica"][4]
        maps, x, gfp_peaks, gev = clustering(data, fs, chs, locs, mode, n_maps, \
                                             doplot=True)
        #print("\t\tK-means clustering of: n = {:d} EEG maps".format(len(gfp_peaks)))
        #print("\t\tMicrostates: {:d} maps x {:d} channels".format(maps.shape[0],maps.shape[1]))
        #print("\t\tGlobal explained variance GEV = {:.2f}".format(gev.sum()))
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (5) basic microstate statistics
        os.system("clear")
        print("\n\t[5] Basic microstate statistics:")
        p_hat = p_empirical(x, n_maps)
        T_hat = T_empirical(x, n_maps)
        print("\n\t\tEmpirical symbol distribution (=RTT):\n")
        for i in range(n_maps):
            print("\t\tp_{:d} = {:.3f}".format(i, p_hat[i]))
        print("\n\t\tEmpirical transition matrix:\n")
        print_matrix(T_hat)

        # PPS (peaks per second)
        pps = len(gfp_peaks) / (len(x)/fs)
        print("\n\t\tGFP peaks per sec.: {:.2f}".format(pps))

        # GEV (global explained variance A-D)
        print("\n\t\tGlobal explained variance (GEV) per map:")
        print("\t\t" + str(gev))
        print("\n\t\tTotal GEV: {:.2f}".format(gev.sum()))
        #raw_input()

        # Cross-correlation between maps
        print("\n\t\tCross-correlation between microstate maps:")
        C = np.corrcoef(maps)
        C_max = np.max(np.abs(C-np.array(np.diag(C)*np.eye(n_maps))))
        print_matrix(C)
        print ("\n\t\tCmax: {:.2f}").format(C_max)

        # (6) entropy of the microstate sequence
        #os.system("clear")
        print("\n\t[6a] Shannon entropy of the microstate sequence:")
        h_hat = H_1(x, n_maps)
        h_max = max_entropy(n_maps)
        print("\n\t\tEmpirical entropy H = {:.2f} (max. entropy: {:.2f})\n".\
              format(h_hat,h_max))
        #raw_input("\n\n\t\tpress ENTER to continue...")

        # (6b) entropy rate of the microstate sequence
        #os.system("clear")
        print("\n\t[6b] Entropy rate of the microstate sequence:")
        kmax = 6
        h_rate, _ = excess_entropy_rate(x, n_maps, kmax, doplot=True)
        h_mc = mc_entropy_rate(p_hat, T_hat)
        print("\n\t\tEmpirical entropy rate h = {:.2f}".format(h_rate))
        print("\t\tTheoretical MC entropy rate h = {:.2f}".format(h_mc))
        raw_input("\n\n\t\tpress ENTER to continue...")

        #''' (7) Markov tests
        os.system("clear")
        print("\n\t[7] Markovianity tests:\n")
        alpha = 0.01
        p0 = testMarkov0(x, n_maps, alpha)
        p1 = testMarkov1(x, n_maps, alpha)
        p2 = testMarkov2(x, n_maps, alpha)
        p_geo_vals = geoTest(x, n_maps, 1000./fs, alpha)
        p_Markov0.append(p0)
        p_Markov1.append(p1)
        p_Markov2.append(p2)
        #p_geo.append(p_geo_vals)
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (8) stationarity test
        os.system("clear")
        print("\n\t[8] Stationarity test:")
        try:
            L = int(raw_input("\t\tenter block size: "))
        except:
            L = 5000
        p3 = conditionalHomogeneityTest(x, n_maps, L, alpha)
        p_stationarity.append(p3)
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (9) symmetry test
        os.system("clear")
        print("\n\t[9] Symmetry test:")
        p4 = symmetryTest(x, n_maps, alpha)
        p_symmetry.append(p4)
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (10) Markov surrogate example
        os.system("clear")
        print("\n\t[10] Synthesize a surrogate Markov chain:")
        x_mc = surrogate_mc(p_hat, T_hat, n_maps, len(x))
        p_surr = p_empirical(x_mc, n_maps)
        T_surr = T_empirical(x_mc, n_maps)
        print("\n\t\tSurrogate symbol distribution:\n")
        for i in range(n_maps):
            print("\t\tp_{:d} = {:.3f}".format(i, p_surr[i]))
        print( "\n\t\tSurrogate transition matrix:\n" )
        print_matrix(T_surr)
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (11) Markov tests for surrogate (sanity check)
        os.system("clear")
        print("\n\t[11] Markovianity tests for surrogate MC:\n")
        p0_ = testMarkov0(x_mc, n_maps, alpha)
        p1_ = testMarkov1(x_mc, n_maps, alpha)
        p2_ = testMarkov2(x_mc, n_maps, alpha)
        p_geo_vals_ = geoTest(x_mc, n_maps, 1000./fs, alpha)
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (12) Stationarity test for surrogate MC
        os.system("clear")
        print("\n\t[12] Stationarity test for surrogate MC:")
        try:
            L = int(raw_input("\t\tenter block size: "))
        except:
            L = 5000
        p3_ = conditionalHomogeneityTest(x_mc, n_maps, L, alpha)
        raw_input("\n\n\t\tpress ENTER to continue...")

        # (13) Symmetry test for surrogate (sanity check)
        os.system("clear")
        print("\n\t[13] Symmetry test for surrogate MC:")
        p4_ = symmetryTest(x_mc, n_maps, alpha)
        raw_input("\n\n\t\tpress ENTER to continue...")
        #'''

        #''' (14) time-lagged mutual information (AIF) empirical vs. Markov model
        os.system("clear")
        print("\n\t[14] Mutual information of EEG microstates and surrogate MC:\n")
        l_max = 100
        #print(argsdict)
        if argsdict['markovsurrogates']:
            n_mc = int(argsdict['markovsurrogates'])
        else:
            n_mc = 10
        aif = mutinf(x, n_maps, l_max)
        jmax, mx_mi = aif_peak1(aif, fs, doplot=True)
        print("\n\t\tAIF 1st peak: index {:d} = {:.2f} ms".format(jmax, mx_mi))

        #print("\n\n\t\tComputing n = {:d} Markov surrogates...".format(n_mc))
        #aif_array = mutinf_CI(p_hat, T_hat, len(x), alpha, n_mc, l_max)
        #pct = np.percentile(aif_array,[100.*alpha/2.,100.*(1-alpha/2.)],axis=0)
        print("\n\n")

        '''
        plt.ioff()
        fig = plt.figure(1, figsize=(20,5))
        fig.patch.set_facecolor('white')
        t = np.arange(l_max)*1000./fs
        plt.semilogy(t, aif, '-sk', linewidth=3, label='AIF (EEG)')
        #plt.semilogy(t, pct[0,:], '-k', linewidth=1)
        #plt.semilogy(t, pct[1,:], '-k', linewidth=1)
        #plt.fill_between(t, pct[0,:], pct[1,:], facecolor='gray', alpha=0.5, label='AIF (Markov)')
        plt.xlabel("time lag [ms]", fontsize=24, fontweight="bold")
        plt.ylabel("AIF [nats]", fontsize=24, fontweight="bold")
        plt.legend(fontsize=22)
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticks(), fontsize=24, fontweight='normal')
        ax.set_yticklabels(ax.get_yticks(), fontsize=24, fontweight='normal')
        plt.tight_layout()
        plt.show()
        '''

    if (len(filenames) > 1):
        os.system("clear")
        print("\n\t[15] Multiple comparisons correction")
        mc_method = 'b'
        if (len(p_Markov0) > 1):
            p_Markov0_mc = multiple_comparisons(p_Markov0, mc_method)
            print("\n\tZero-order Markov test: " + str(p_Markov0_mc))
        if (len(p_Markov1) > 1):
            p_Markov1_mc = multiple_comparisons(p_Markov1, mc_method)
            print("\n\tFirst-order Markov test: " + str(p_Markov1_mc))
        if (len(p_Markov2) > 1):
            p_Markov2_mc = multiple_comparisons(p_Markov2, mc_method)
            print("\n\tZero-order Markov test: " + str(p_Markov0_mc))
        if (len(p_stationarity) > 1):
            p_stationarity_mc = multiple_comparisons(p_stationarity, mc_method)
            print("\n\tStationarity test: " + str(p_stationarity_mc))
        if (len(p_symmetry) > 1):
            p_symmetry_mc = multiple_comparisons(p_symmetry, mc_method)
            print("\n\tSymmetry test: " + str(p_symmetry_mc))

    print("\n\n\t\t--- Tutorial completed ---")


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    main()
