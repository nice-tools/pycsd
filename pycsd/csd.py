# PyCSD
#
# Copyright 2003-2010 Jürgen Kayser <rjk23@columbia.edu>
# Copyright 2017 Federico Raimondo <federaimondo@gmail.com> and
#                Denis A. Engemann <dengemann@gmail.com>
#
# The following code is a derivative work of the code from the CSD Toolbox,
# which is licensed GPLv3. This code therefore is also licensed under the terms
# of the GNU Public License, version 3.
#
# The original CSD Toolbox can be find at
# http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox/

import numpy as np

import os
import os.path as op

from numpy.polynomial.legendre import legval
from scipy.linalg import inv

import mne
from mne import pick_types, pick_info
from mne.utils import logger
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.parallel import parallel_func

from packaging import version

if version.parse(mne.__version__) >= version.parse('0.20.dev0'):
    _mne_20_api = True
    logger.info('Using MNE with API > 0.19')
else:
    _mne_20_api = False
    logger.info('Using MNE with API <= 0.19')

def _get_montage_pos(montage, picks):
    if _mne_20_api is True:
        pos = np.array([x['r'] for x in montage.dig])
    else:
        pos = montage.pos
    return pos[picks]

def _read_csd_montage(fname):
    if _mne_20_api is True:
        montage = mne.channels.read_custom_montage(fname, head_size=None)
    else:
        montage = mne.channels.read_montage(fname)
    return montage

def _extract_positions(inst, picks):
    """Aux function to get positions via Montage
       The mongate is specified in the info['description'] field
    """
    if '/' in inst.info['description']:
        system, n_channels = inst.info['description'].split('/')
    else:
        system = 'default'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if system == 'egi':
        if n_channels == '256':
            logger.info('Using EGI 256 locations for CSD')
            montage = _read_csd_montage(
                op.join(dir_path, 'templates/EGI_256.csd'))
        elif n_channels == '128':
            logger.info('Using EGI 128 locations for CSD')
            montage = _read_csd_montage(
                op.join(dir_path, 'templates/EGI_128.csd'))
        else:
            raise ValueError('CSD Lookup not defined for egi/{}'.format(n_eeg))
    else:
        logger.info('Using 10-5 locations for CSD')
        montage = _read_csd_montage(
            op.join(dir_path, 'templates/standard_10-5.csd'))
    pos_picks = [montage.ch_names.index(x) for x in inst.ch_names]
    pos = _get_montage_pos(montage, pos_picks)
    return pos[picks if picks is not None else Ellipsis]


def _calc_g(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline.
    num_lterms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) / (n ** stiffnes * (n + 1) ** stiffnes * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _calc_h(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline. Also referred to as `m`.
    num_lterms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) /
               (n ** (stiffnes - 1) * (n + 1) ** (stiffnes - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _prepare_G(G, lambda2):
    # regularize if desired
    if lambda2 is None:
        lambda2 = 1e-5

    G.flat[::len(G) + 1] += lambda2
    # compute the CSD
    Gi = inv(G)

    TC = Gi.sum(0)
    sgi = np.sum(TC)  # compute sum total

    return Gi, TC, sgi


def _compute_csd(data, G_precomputed, H, head):
    """compute the CSD"""
    n_channels, n_times = data.shape
    mu = data.mean(0)[None]
    Z = data - mu
    X = np.zeros_like(data)
    head **= 2

    Gi, TC, sgi = G_precomputed

    Cp2 = np.dot(Gi, Z)
    c02 = np.sum(Cp2, axis=0) / sgi
    C2 = Cp2 - np.dot(TC[:, None], c02[None, :])
    X = np.dot(C2.T, H).T / head
    return X


def epochs_compute_csd(inst, picks=None, g_matrix=None, h_matrix=None,
                       lambda2=1e-5, head=1.0, lookup_table_fname=None,
                       n_jobs=1, copy=True):
    """ Current Source Density (CSD) transformation

    Transormation based on spherical spline surface Laplacian as suggested by
    Perrin et al. (1989, 1990), published in appendix of Kayser J, Tenke CE,
    Clin Neurophysiol 2006;117(2):348-368)

    Implementation of algorithms described by Perrin, Pernier, Bertrand, and
    Echallier in Electroenceph Clin Neurophysiol 1989;72(2):184-187, and
    Corrigenda EEG 02274 in Electroenceph Clin Neurophysiol 1990;76:565.

    Parameters
    ----------
    inst : instance of Epochs or Evoked
        The data to be transformed.
    picks : np.ndarray, shape (n_channels,) | None
        The picks to be used. Defaults to None.
    g_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the g_function will be computed from the data (default).
    h_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the h_function will be computed from the data (default).
    lambda2 : float
        Regularization parameter, produces smoothnes. Defaults to 1e-5.
    head : float
        The head radius (unit sphere). Defaults to 1.
    n_jobs : int
        The number of processes to run in parallel. Note. Only used for
        Epochs input. Defaults to 1.
    copy : bool
        Whether to overwrite instance data or create a copy.

    Returns
    -------
    inst_csd : instance of Epochs or Evoked
        The transformed data. Output type will match input type.
    """

    if copy is True:
        out = inst.copy()
    else:
        out = inst
    if picks is None:
        picks = pick_types(inst.info, meg=False, eeg=True, exclude='bads')
    if len(picks) == 0:
        raise ValueError('No EEG channels found.')

    if ((g_matrix is None or h_matrix is None)):
        pos = _extract_positions(inst, picks=picks)

    G = _calc_g(np.dot(pos, pos.T)) if g_matrix is None else g_matrix
    H = _calc_h(np.dot(pos, pos.T)) if h_matrix is None else h_matrix
    G_precomputed = _prepare_G(G, lambda2)

    if isinstance(out, BaseEpochs):
        n_jobs = min(len(inst), n_jobs)
        logger.info('Using {} jobs'.format(n_jobs))
        parallel, my_csd, _ = parallel_func(_compute_csd, n_jobs)
        data = np.asarray(parallel(my_csd(e[picks],
                                   G_precomputed=G_precomputed,
                                   H=H, head=head) for e in out))
        out.preload = True
        out._data = data
    elif isinstance(out, Evoked):
        out.data = _compute_csd(out.data[picks], G_precomputed=G_precomputed,
                                H=H, head=head)
    pick_info(out.info, picks, copy=False)
    return out
