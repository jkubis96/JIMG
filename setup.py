from setuptools import setup, find_packages

VERSION = '1.4.5' 
DESCRIPTION = 'operetta_tool'
LONG_DESCRIPTION = 'The Operetta_tool is a python library created for handling and annotation images from the Opera Phenix platform used for ML / AI applications. Instructions for use on github [https://github.com/jkubis96/Operetta_tool] '

# Setting up
setup(
        name="Operetta_tool", 
        version=VERSION,
        author="Jakub Kubis",
        author_email="jbiosystem@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=['operetta'],
        include_package_data=True,
        install_requires=['regex', 'pandas', 'numpy', 'more-itertools', 'opencv-python', 'matplotlib', 'Pillow', 'h5py', 'tqdm', 'tifffile', 'joblib', 'tk', 'scikit-image'],       
        keywords=['python', 'opera', 'images', 'annotation', 'AI', 'cv', 'perkin'],
        license = 'MIT',
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
        ],
        python_requires='>=3.6',
)


