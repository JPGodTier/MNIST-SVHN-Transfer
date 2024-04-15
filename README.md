# MSIA-MNIST-SVHN-Transfer
MNIST to SVHN Domain Adaptation for Computer Vision

## Getting Started

### Dependencies

Refer to `requirements.txt` for a full list of dependencies.

### Installing

#### For Users:

* To install our project: 

```
git clone https://github.com/JPGodTier/MNIST-SVHN-Transfer
cd MNIST-SVHN-Transfer
pip install .
```

#### For Developers/Contributors:

If you're planning to contribute or test the latest changes, you should first set up a virtual environment and then install the package in "editable" mode. This allows any changes you make to the source files to immediately affect the installed package without requiring a reinstall.

* Clone the repository:

```
git clone https://github.com/JPGodTier/MNIST-SVHN-Transfer
cd MNIST-SVHN-Transfer
```

* Set up a virtual environment:

```
python3 -m venv mstrans_env
source mstrans_env/bin/activate  # On Windows, use: mstrans_env\Scripts\activate
```

* Install the required dependencies:

```
pip install -r requirements.txt
```

* Install the project in editable mode:

```
pip install -e . 
```

### Executing program

Launch the different model Runners:  
```
python3 bin/CyCADARunner.py
python3 bin/DiscrepancyClassifierRunner.py
python3 bin/PLRunner.py
```

## Authors

* Paul Aristidou
* Olivier Lapabe-Goastat

## Version History

* **1.0.0** - Initial release