from pathlib import Path

from setuptools import setup, find_packages


if __name__ == '__main__':

    with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name="recommendation-models",
        version="0.1",
        description="PyTorch implementation of Recommendation Models",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/rixwew/torchfm",
        packages=find_packages(exclude=["examples", "docs", "test", "data"]),
    )
