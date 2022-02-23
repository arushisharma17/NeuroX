
# Contributing to NeuroX

Welcome, and thank you for your interest in improving NeuroX. This document provides a high-level overview of the contribution process. For any clarifications and/or concerns, feel free to open an issue in the GitHub issue tracker.

NeuroX is a toolkit which is evolving and growing quite quickly, so it is important to share your intentions before you start working on something. This is to make sure that your effort is not being duplicated elsewhere. We expect contributions to be of several kinds like:

1. Bug fixes (existing in the tracker or new)
2. Quality-of-Life Improvements (this may mean better wrappers around existing code, better documentation, improved/additional tests, example notebooks)
3. New algorithms for neuron discovery and analysis (usually an implementation of an existing paper)
4. New general algorithms (for things like processing data, filtering, balancing, performing analysis, evaluation, etc)

There may be other types of contributions that we may have missed, so feel free to open an issue. For an idea within the above categorization, please mention the category in your issue, along with supporting code/documents such as a minimal reproducible example for bug fixes, or links to papers for new algorithms. Once you have received a go-ahead, implement your feature, document your code and add tests before submitting a pull request to the repository.


## Development Process

### Code Style

NeuroX aims to keep a consistent style, and is enforcing `black` for all future contributions.


### Unit Tests

To run the unit tests, you can use python's `unittest` module:

```bash
python -m unittest
```


### Documentation
The following command will build the documentation locally and launch the page in your browser:
```bash
./generate_docs.sh
```

## Pull Requests
We actively welcome pull requests (and appreciate them to be pre-discussed in an issue).

1. Fork the repo and create your branch from `master`.
2. Implement your bugfix/feature in the new branch
3. Add tests for any new/modified code
4. Ensure the entire test suite passes
5. Add/Modify documentation for any new/modified code


## Issues

We use GitHub issues to track all bugs and discussion, please be as specific as possible in your issue.


## License

By contributing to NeuroX, you agree that your contributions will be licensed under BSD 3-Clause License.