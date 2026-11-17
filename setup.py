"""Compatibility shim.

Modern installs read ``pyproject.toml`` (PEP 621, setuptools >= 64). Some older
toolchains still ship setuptools 58, which ignores that table — this file keeps
``python setup.py develop`` and legacy ``pip`` working there.

Nothing is defined twice: values here mirror pyproject.toml.
"""

from setuptools import find_packages, setup

setup(
    name="reforge",
    version="0.3.1",
    description="Reverse-engineer any AI image/video into a reproducible recipe.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(include=["reforge", "reforge.*"]),
    package_data={"reforge": ["static/*.html"]},
    python_requires=">=3.9",
    install_requires=["numpy>=1.20", "pillow>=9.0"],
    extras_require={
        "controls": ["opencv-python-headless>=4.8", "onnxruntime>=1.16"],
        "full": ["opencv-python-headless>=4.8", "onnxruntime>=1.16", "torch"],
        "space": ["gradio>=4.44"],
        "dev": ["pytest", "build", "torch", "onnx"],
    },
    entry_points={"console_scripts": ["reforge=reforge.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
    ],
)
