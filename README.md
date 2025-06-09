# Kibble-Zurek and Landau-Zener Workshop

A repository of code to demonstrate simple Kibble-Zurek and Landau-Zener
experiments compatible with those published in these papers:

* [Quantum critical dynamics in a 5,000-qubit programmable spin glass](https://doi.org/10.1038/s41586-023-05867-2) (Figure 2C)
* [Coherent quantum annealing in a programmable 2,000 qubit Ising chain](https://doi.org/10.1038/s41567-022-01741-6) (Figure 2A)

## Installation

You can run this example without installation in cloud-based IDEs that support
the [Development Containers specification](https://containers.dev/supporting)
(aka "devcontainers") such as GitHub
[Codespaces](https://github.com/features/codespaces).

Note that a Leap account's token should be made available to the Ocean
software. For workshop participants this is achieved by typing and entering the
workshop token when requested.

### Codespaces (Online IDE)

The Codespaces free-tier allocation associated to your GitHub account is
sufficient to run the experiments.

Click on the code tab (top right) and select codespaces.

Follow the instructions printed to the terminal to set up your leap credentials.

```bash
dwave setup --oob
```
### Local IDE

Clone the repo. If you are cloning the repo to your local system, working in a
[virtual environment](https://docs.python.org/3/library/venv.html) is
recommended. Python 3.9 to 3.13 versions are supported.

For development environments that do not support `devcontainers`, install
requirements:

```bash
pip install -r requirements.txt
```

```bash
dwave setup --oob
```

## Usage

The experiments are run with defaults by executing the following command from
the terminal:

```bash
python3 main.py
```

### Inputs

To see configuration options
```bash
python3 main.py --help
```

### Outputs

The code prints to file and/or displays embeddings and a figure demonstrating
the power-law or exponential scaling associated to Kibble-Zurek and Landau-Zener phenomena respectively.

## Problem Description

See Figure 2C of the cited papers to understand details of the experiment.

## Code Overview

* Define a target problem
* Connect to a solver
* Construct mapping(s) from the problem to the solver (embedding)
* Submit QPU instructions varying by fast-anneal time collecting data
* Plot the data

## References

* [Quantum critical dynamics in a 5,000-qubit programmable spin glass](https://doi.org/10.1038/s41586-023-05867-2)
* [Coherent quantum annealing in a programmable 2,000 qubit Ising chain](https://doi.org/10.1038/s41567-022-01741-6)

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
