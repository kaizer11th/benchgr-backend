from setuptools import setup, find_packages

setup(
    name="benchgr-agent",
    version="1.0.0",
    description="BenchGR GPU benchmark agent — test your GPU and submit to the leaderboard",
    author="BenchGR",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "rich>=13.7.0",
        "click>=8.1.7",
        "pynvml>=11.5.0",
        "numpy>=1.26.0",
    ],
    extras_require={
        "full": [
            "torch>=2.1.0",
            "transformers>=4.40.0",
            "diffusers>=0.27.0",
            "accelerate>=0.29.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "benchgr=benchgr_agent.cli:main",
        ],
    },
    python_requires=">=3.9",
)
