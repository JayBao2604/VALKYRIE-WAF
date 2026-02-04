"""
DEG-WAF: Deep Learning Web Application Firewall
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="deg-waf",
    version="0.1.0",
    author="DEG-WAF Team",
    description="Deep Learning Web Application Firewall for attack payload generation and detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deg-waf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "deg-waf-train=scripts.train:main",
            "deg-waf-generate=scripts.generate_payloads:main",
        ],
    },
    include_package_data=True,
    package_data={
        "deg_waf": ["py.typed"],
    },
)
