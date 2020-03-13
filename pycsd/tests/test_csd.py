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


import os.path as op

import numpy as np
from scipy import io as sio

from numpy.testing import assert_almost_equal
from nose.tools import assert_equal
import mne
from mne import create_info
from mne.io import RawArray

from packaging import version

from pycsd import epochs_compute_csd

from pycsd.csd import _calc_g
from pycsd.csd import _calc_h
from pycsd.csd import _compute_csd
from pycsd.csd import _extract_positions
from pycsd.csd import _prepare_G

n_epochs = 3
n_channels = 6
n_samples = n_epochs * 386
mat_contents = sio.loadmat(
    op.join(op.realpath(op.dirname(__file__)), 'data', 'test-eeg.mat'))
data = mat_contents['data'][:n_channels, :n_samples] * 1e-7
sfreq = 250.
ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)]
ch_names += ['STI 014']
ch_types = ['eeg'] * n_channels
ch_types += ['stim']
data = np.r_[data, data[-1:]]
data[-1].fill(0)
info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
raw = RawArray(data=data, info=info)

if version.parse(mne.__version__) > version.parse('0.19'):
    montage = mne.channels.make_standard_montage(
        'GSN-HydroCel-257', head_size=0.085)
else:
    montage = mne.channels.read_montage('GSN-HydroCel-257')

raw.set_montage(montage)
raw.info['description'] = 'egi/256'

triggers = np.arange(50, n_epochs*386, 386)

raw._data[-1].fill(0.0)
raw._data[-1, triggers] = [10] * n_epochs

events = mne.find_events(raw)
event_id = {
    'foo': 10,
}
epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                    preload=True, reject=None, picks=None,
                    baseline=(None, 0), verbose=False)
epochs.drop_channels(['STI 014'])
epochs.info['description'] = 'egi/256'
picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')

csd_data = sio.loadmat(
    op.join(op.realpath(op.dirname(__file__)), 'data', 'test-eeg-csd.mat'))


def test_csd_core():
    """Test G, H and CSD against matlab CSD Toolbox"""
    positions = _extract_positions(epochs, picks)
    cosang = np.dot(positions, positions.T)
    G = _calc_g(cosang)
    assert_almost_equal(G, csd_data['G'], 6)
    H = _calc_h(cosang)
    assert_almost_equal(H, csd_data['H'], 5)
    G_precomputed = _prepare_G(G.copy(), lambda2=1e-5)
    for i in range(n_epochs):
        csd_x = _compute_csd(
            epochs._data[i], G_precomputed=G_precomputed, H=H, head=1.0)
        assert_almost_equal(csd_x, csd_data['X'][i], 4)

    assert_almost_equal(G, csd_data['G'], 6)
    assert_almost_equal(H, csd_data['H'], 5)


def test_compute_csd():
    """Test epochs_compute_csd function"""
    csd_epochs = epochs_compute_csd(epochs)
    assert_almost_equal(csd_epochs._data, csd_data['X'], 7)

    csd_evoked = epochs_compute_csd(epochs.average())
    assert_almost_equal(csd_evoked.data, csd_data['X'].mean(0), 7)
    assert_almost_equal(csd_evoked.data, csd_epochs._data.mean(0), 7)

    csd_epochs = epochs_compute_csd(epochs, picks=picks[:4])
    assert_equal(csd_epochs._data.shape, (n_epochs, 4, 386))
    assert_equal(csd_epochs.ch_names, epochs.ch_names[:4])


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
