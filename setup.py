from setuptools import find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "Python package for simulating molecular trajectories in CeNTREX"

# Setting up
setup(
    name="trajectories",
    version=VERSION,
    author="Oskari Timgren",
    author_email="<oskari.timgren@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "joblib", "h5py"],
    keywords=["python",],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
