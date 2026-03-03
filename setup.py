from setuptools import setup, find_packages

setup(
    name="medcore",
    version="1.1.0",
    description="Medical imaging utilities based on SimpleITK",
    long_description=open("README.md", "r", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="yongho choi",
    author_email="yhchoi@hutom.co.kr",
    maintainer="DATA",
    maintainer_email="dm@hutom.co.kr",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pydicom>=3.0",
        "SimpleITK>=2.4",
    ],
    include_package_data=True,
    zip_safe=False,
)
