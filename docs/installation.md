# Installation

## Required Dependencies

- Python (3.11 or later)
- [numpy](https://www.numpy.org/) (2.0.0 or later)
- [pandas](https://pandas.pydata.org/) (2.1.0 or later)
- [SciPy](https://scipy.org/) (1.14.0 or later)
- [scikit-learn](https://scikit-learn.org/) (1.5.1 or later)
- [matplotlib](https://matplotlib.org/) (3.9.0 or later)
- [loguru](https://github.com/Delgan/loguru) (0.7.2 or later)
- [notebook](https://jupyter-notebook.readthedocs.io/) (7.4.4 or later)

## Installation Methods

It's recommended to install ``statista`` in a virtual environment to avoid conflicts with your system's Python packages.

### Conda

The easiest way to install ``statista`` is using the ``conda`` package manager. ``statista`` is available in the
[conda-forge](https://conda-forge.org/) channel. To install, use the following command:

```bash
conda install -c conda-forge statista
```

This will install `statista` with all dependencies including Python, numpy, scipy, scikit-learn, and other required packages.
If this works, you can skip the rest of the installation instructions.


### Installing Python

The main dependencies for statista are Python 3.11+ and the scientific Python stack.

For Python, we recommend using the Anaconda Distribution for Python 3, which is available
for download from [https://www.anaconda.com/download/](https://www.anaconda.com/download/). The installer gives the option to
add `python` to your `PATH` environment variable. We will assume in the instructions
below that it is available in the path, such that `python`, `pip`, and `conda` are
all available from the command line.

Note that there is no hard requirement specifically for Anaconda's Python, but it
makes installation of required dependencies easier using the conda package manager.

### Install in a New Conda Environment

The easiest and most robust way to install statista is by installing it in a separate
conda environment. You can create a new environment with the required dependencies and then install statista.

Run these commands to create a new environment with the necessary dependencies:

```bash
conda create -n statista python=3.11
conda activate statista
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib loguru
```

This creates a new environment with the name `statista` and installs the required dependencies.
To activate this environment in a session, run:

```bash
conda activate statista
```

For the installation of statista there are two options (from the Python Package Index (PyPI)
or from GitHub):

1. To install the latest release of statista from PyPI:

```bash
pip install statista
```

2. To install a specific version (e.g., 0.6.1):

```bash
pip install statista==0.6.1
```


### From Sources

The sources for statista can be downloaded from the [GitHub repository](https://github.com/serapeum-org/statista).

You can either clone the public repository:

```bash
git clone https://github.com/serapeum-org/statista.git
```

Or download the [tarball](https://github.com/serapeum-org/statista/tarball/main):

```bash
curl -OJL https://github.com/serapeum-org/statista/tarball/main
```

Once you have a copy of the source, you can install it with:

```bash
python -m pip install .
```

To install directly from GitHub (from the HEAD of the main branch):

```bash
pip install git+https://github.com/serapeum-org/statista.git
```

Or from GitHub for a specific release (e.g., 0.6.1):

```bash
pip install git+https://github.com/serapeum-org/statista.git@0.6.1
```

Now you should be able to start this environment's Python with `python` and try
`import statista` to see if the package is installed.

More details on how to work with conda environments can be found in the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Development Installation

If you are planning to make changes and contribute to the development of statista, it is
best to make a git clone of the repository and do an editable install. This will not move a copy to your Python installation directory, but
instead create a link in your Python installation pointing to the folder you installed
it from, so any changes you make there are directly reflected in your install.

```bash
git clone https://github.com/serapeum-org/statista.git
cd statista
conda activate statista  # or your preferred environment
pip install -e .
```

For development, you might also want to install the development dependencies:

```bash
pip install -e ".[dev]"
```

Alternatively, if you want to avoid using `git` and simply want to test the latest
version from the `main` branch, you can download a
[zip archive from GitHub](https://github.com/serapeum-org/statista/archive/main.zip).

### Install Using Pip

Besides the recommended conda environment setup described above, you can also install
statista with `pip`. For the scientific Python dependencies, you might want to use the conda package manager first:

```bash
conda install numpy scipy scikit-learn matplotlib pandas
pip install loguru notebook
```

Then install statista with pip:

```bash
pip install statista
```

Or install a specific version (e.g., 0.6.1):

```bash
pip install statista==0.6.1
```

You can check [libraries.io](https://libraries.io/github/serapeum-org/statista) to see the latest versions of the dependencies.

## Verifying the Installation

To check if the installation is successful, run the following command in your Python environment:

```python
import statista
print(statista.__version__)
```

You can also try running one of the example scripts from the examples directory:

```bash
python examples/extreme-value-statistics.py
```
