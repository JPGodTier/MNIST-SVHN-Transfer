from setuptools import setup, find_packages

setup(
    name="MNIST-SVHN-Transfer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "contourpy == 1.2.0",
        "cycler == 0.12.1",
        "filelock == 3.13.1",
        "fonttools == 4.49.0",
        "fsspec == 2024.2.0",
        "Jinja2 == 3.1.3",
        "kiwisolver == 1.4.5",
        "MarkupSafe == 2.1.5",
        "matplotlib == 3.8.3",
        "mpmath == 1.3.0",
        "networkx == 3.2.1",
        "numpy == 1.26.4",
        "packaging == 24.0",
        "pillow == 10.2.0",
        "pyparsing == 3.1.2",
        "python-dateutil == 2.9.0.post0",
        "scipy == 1.12.0",
        "six == 1.16.0",
        "sympy == 1.12",
        "torch == 2.2.1+cu118",
        "torchaudio==2.2.1+cu118"
        "torchvision == 0.17.1",
        "tqdm == 4.66.2",
        "typing_extensions == 4.10.0",
        "colorama == 0.4.6"
    ]
)
