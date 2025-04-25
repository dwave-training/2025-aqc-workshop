# Kibble-Zurek and Landau-Zener journal experiments in ocean-sdk

A repository of code to demonstrate simple Kibble-Zurek and Landau-Zener
experiments compatible with

* https://doi.org/10.1038/s41586-023-05867-2 (Figure 2C)
* https://doi.org/10.1038/s41567-022-01741-6 (Figure 2A)

Attendees are encouraged to run and extend these examples. See also our
collection of demos such as the zero-noise extrapolation demo:

and other instructive codebases such as the shimming-tutorial:


![D-Wave Logo](dwave_logo.png)

## Usage

A simple command that runs your program. For example,

```bash
python aqc_main.py
```

### Inputs
First draft. No inputs.

### Outputs
The code displays embeddings and plots demonstrating power-law or exponential scaling of energy.

## Problem Description 

The first example reproduces published Kibble-Zurek phenomena.

The second example reproduces published Landau-Zener phenomena.

## Code Overview

A general overview of how the code works.

We prefer descriptions in bite-sized bullet points:

* Define a target problem
* Connect to a solver
* Construct mapping(s) from the problem to the solver (embedding)
* Submit QPU instructions varying by fast-anneal time collecting data
* Plot the data

## Code Specifics

Ocean-sdk is under development, please contribute code and requests, and report issues. 

## References

* https://doi.org/10.1038/s41586-023-05867-2
* https://doi.org/10.1038/s41567-022-01741-6

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
