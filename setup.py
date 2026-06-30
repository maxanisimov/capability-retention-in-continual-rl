from setuptools import find_packages, setup


setup(
    name="capability-retention-in-continual-rl",
    version="0.1.0",
    description="Capability retention in continual RL with Rashomon-set tooling.",
    python_requires=">=3.10",
    package_dir={"": "core", "projects": "projects"},
    packages=find_packages(
        where="core",
        include=[
            "abstract_gradient_training*",
            "certified_continual_learning*",
            "configs*",
            "provably_safe_policy_optimisation*",
            "safe_rl_baselines*",
            "src*",
        ],
    )
    + find_packages(
        where=".",
        include=["projects*"],
    ),
    include_package_data=True,
    install_requires=[
        "cooper-optim>=1.0.1",
        "numpy>=2.2.0",
        "scipy>=1.15.0",
        "torch>=2.0.0",
        "tqdm>=4.0.0",
    ],
    extras_require={
        "rl": [
            "gymnasium>=1.0.0",
            "highway-env>=1.10.0",
            "pygame>=2.6.1",
            "sb3-contrib>=2.7.0",
            "stable-baselines3>=2.7.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "pandas>=2.0.0",
            "seaborn>=0.13.0",
        ],
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
            "ruff>=0.5.0",
        ],
    },
)
