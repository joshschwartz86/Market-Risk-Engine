from setuptools import setup, find_packages

setup(
    name="market_risk_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "lxml>=4.9.0",
        "pydantic>=2.5.0",
    ],
    python_requires=">=3.10",
)
