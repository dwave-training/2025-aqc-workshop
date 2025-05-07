# Kibble-Zurek and Landau-Zener experiments in ocean-sdk

** For unknown reasons the requirements.txt install correctly for codespaces but not locally. For now, install the required packages individually if you are not using codespaces. This should be a non-issue once ocean-sdk is updated (before the workshop). **

A repository of code to demonstrate simple Kibble-Zurek and Landau-Zener
experiments compatible with

* https://doi.org/10.1038/s41586-023-05867-2 (Figure 2C)
* https://doi.org/10.1038/s41567-022-01741-6 (Figure 2A)

![D-Wave Logo](dwave_logo.png)

## Usage 

### Codespaces usage
(codespaces)

The codespaces free-tier allocation associated to your github account is sufficient to run the experiments.

Click on the code tab (top right) and select codespaces.

Set up your leap credentials:
```bash
dwave setup --oob
```

The experiment is run under defaults as:

```bash
python3 main.py
```


## Local usage
Clone the repo.

In a python3 environment (3.9 to 3.13) install the requirements:

```bash
pip install -r requirements.txt
```

Note that Leap token should be made available to ocean-sdk. This might typicaly be achieved by typing and entering the workshop token.

```bash
dwave setup --oob
```

The experiment is run under defaults as

```bash
python3 main.py
```

### Inputs

To see configuration options
```bash
python3 main.py --help
```

### Outputs

The code prints to file and/or displays embeddings and a figure demonstrating the power-law or exponential scaling associated to Kibble-Zurek and Landau-Zener phenomena respectively.

## Problem Description 

See Figure 2C of the cited papers to understand details of the experiment.

## Code Overview

* Define a target problem
* Connect to a solver
* Construct mapping(s) from the problem to the solver (embedding)
* Submit QPU instructions varying by fast-anneal time collecting data
* Plot the data

## References

* https://doi.org/10.1038/s41586-023-05867-2
* https://doi.org/10.1038/s41567-022-01741-6

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
