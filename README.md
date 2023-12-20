# iDDN
We developed an efficient and accurate multi-omics differential network analysis tool – integrative Differential Dependency Networks (iDDN).
iDDN is capable of jointly learning sparse common and rewired network structures, which is especially useful for genomics, proteomics, and other biomedical studies.
This repository provides the source code and examples of using DDN.

## Installation
### Option 1: install into a new Conda environment using pip
One way is to install DDN into a new Conda environment. To create and activate an environment named `iddn`, run this:
```bash
conda create -n iddn python=3.11
conda activate iddn
```
Python 3.12 may have some issue with Numba.

DDN 3.0 can then be installed with the followin command.
```bash
pip install iddn
```
<!-- ```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple iddn
``` -->

### Option 2: install into an existing Conda environment
If you want to install DDN into an existing Conda environment, it is suggested to install dependencies from Conda first.

First we need to install some common dependencies.
```bash
$ conda install -c conda-forge numpy scipy numba networkx matplotlib jupyter scipy pandas scikit-learn
```

Then run
```bash
pip install iddn
```

Alternatively, you can clone the repository, or just download or unzip it. Then we can install DDN 3.0.
```bash
$ pip install ./
```
Or you may want to install it in development mode.
```bash
$ pip install -e ./
```

## Usage

This toy example generates two random datasets, and use estimate to estimate two networks, one for each dataset.
```python
import numpy as np
from iddn import iddn
dat1 = np.random.randn(1000, 10)
dat2 = np.random.randn(1000, 10)
networks = iddn.iddn(dat1, dat2, lambda1=0.3, lambda2=0.1)
```

For more details and examples, check the [documentation](https://iddn.readthedocs.io/en/latest/), which includes three tutorials and the API reference.
The tutorials can also be found in the `docs/notebooks` folder.

## Tests

To run tests, go to the folder of DDN3 source code, then run `pytest`.
```bash
pytest tests
```
It will compare output of DDN with reference values. It tests DDN with various acceleration strategies.

## Contributing

Please report bugs in the issues. 
You may also email the authors directly: Yingzhou Lu (lyz66@vt.edu), Yizhi Wang (yzwang@vt.edu), or Yue Wang (yuewang@vt.edu).
If you are interested in adding features or fixing bug, feel free to contact us.

## License

The `iddn` package is licensed under the terms of the MIT license.

## Citations

[1] Zhang, Bai, and Yue Wang. "Learning structural changes of Gaussian graphical models in controlled experiments." arXiv preprint arXiv:1203.3532 (2012).
