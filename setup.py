from setuptools import find_packages, setup

setup(
    name="tona-ai",
    version="0.0.0",
    description="A simple package to work with AI",
    author="Tonaxis",
    # author_email="",
    url="https://github.com/tonaxis/tona-ai",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests")),
    python_requires=">=3.4",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
