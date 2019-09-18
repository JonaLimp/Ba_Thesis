import setuptools


with open("README.md") as file:
    long_description = file.read()

with open("requirements.txt") as file:
    install_requires = [line.split("#egg=")[-1] for line in file.read().splitlines()]

setuptools.setup(
    name="code",
    author="Jonas Limpert",
    author_email="jlimpoert@uni-osnabrueck.de",
    description="Jonas Bachelor thesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnasduke/Ba_Methods_CNN",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
