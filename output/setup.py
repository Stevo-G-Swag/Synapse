from setuptools import setup, find_packages

setup(
    name="project_name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "flask",
        "sqlalchemy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of the project",
    keywords="sample, setuptools, development",
    url="https://github.com/yourusername/project_name",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
