from setuptools import setup, find_packages

setup(
    name='Generational Pruning',
    version='0.1.0',
    author="Greta Tuckute, Klemen Kotar",
    description="Official implementation of Model Connectomes: A Generational Approach to Data-Efficient Language Models",
    install_requires=[
        'torch',
        'scipy',
        'tqdm',
        'wandb',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'transformers',
        'tiktoken',
        'huggingface_hub',
    ],
)
