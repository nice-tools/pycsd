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

from os import path as op

import setuptools
from numpy.distutils.core import setup

# get the version
version = None
with open(op.join('pycsd', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'pycsd'
DESCRIPTION = 'CSD Toolbox (python).'
MAINTAINER = '@fraimondo and @dengemann'
MAINTAINER_EMAIL = 'federaimondo@gmail.com'
URL = 'https://github.com/nice-tools/pycsd'
LICENSE = 'GPLv3'
DOWNLOAD_URL = 'https://github.com/nice-tools/pycsd'
VERSION = version


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['pycsd'],
          package_data={'pycsd': ['templates/*.csd']})
