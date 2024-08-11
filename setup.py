from setuptools import setup, find_packages

VERSION = '2.1.2' 
DESCRIPTION = 'JIMG'
LONG_DESCRIPTION = 'This library was created for handling high-resolution images from the Opera Phenix Plus High-Content Screening System, including operations such as concatenating raw series of images, z-projection, channel merging, image resizing, etc. Additionally, we have included options for annotating specific parts of images and selecting them for further analysis, for example, teaching ML/AI algorithms. Certain elements of this tool can be adapted for data analysis and annotation in other imaging systems. For more information, please feel free to contact us!'


# Setting up
setup(
        name="JIMG", 
        version=VERSION,
        author="Jakub Kubis",
        author_email="jbiosystem@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=['JIMG'],
        include_package_data=True,
        install_requires=['numpy', 'pandas', 'opencv-python', 'matplotlib', 'tifffile', 'joblib', 'Pillow'],       
        keywords=['python', 'opera', 'pheonix' 'images', 'annotation', 'AI', 'cv', 'perkin', 'high-resolution'],
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


