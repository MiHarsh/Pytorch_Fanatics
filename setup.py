from setuptools import setup, Extension
from setuptools import find_packages


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="pytorch_fanatics",
        version='0.2.7',
        description="A new library for Computer Vision",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Harsh Mishra and Tanish Gupta",
        author_email="harshm17172612@gmail.com",
        url="https://github.com/MiHarsh/pytorch_fanatics",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        platforms=["linux", "unix"],
        python_requires=">3.5.2",
        install_requires=["scikit-learn>=0.22.1",],
    )
