from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "fast_ska",
    version = "0.9.2",
    description='A fast Cython implementation of the "Streaming K-mer Assignment" algorithm initially described in Lambert et al. 2014 (PMID: 24837674)',
    url = 'https://github.com/marvin-jens/fast_ska',
    author = 'Marvin Jens',
    author_email = 'mjens@mit.edu',
    license = 'MIT',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords = 'rna rbns k-mer kmer statistics biology bioinformatics',

    requires=['cython','numpy'],
    scripts=['ska'],
    ext_modules = cythonize("ska_kmers.pyx")
)
