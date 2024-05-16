# Welcome to contributing to Vibravox

If you're reading this, you're probably interested in contributing to Vibravox. We're happy to have you here!

## How to get started?

This repo is meant to be a collaborative project on several tasks related to speech processing recorded by body-conducted microphones. Here is the list of currently supported tasks:

| Task Tag | Description          |
|----------|----------------------|
| `bwe`    | Bandwidth Extension  |
| `stp`    | Speech to Phoneme    |
| `stt`    | Speech to Text       |
| `spkv`   | Speaker Verification |


### Adding a new model to an existing task
If you're using the Vibravox dataset, you can add your model to one of the supported tasks. Here's how to do it:
- The entry point of your code should be the `run.py` script.
- You must use the lightning datamodule corresponding to your task.
- You can create a new lightning module, some torch_modules or any other utils you need.
- For every created class, you should add the corresponding yaml file in the `configs` folder.
- Finally, your method should be run with the following command:
```python run.py lightning_datamodule=<task_tag> lightning_module=<your_model> <other_args>```

### Adding a new task
Please open an issue to discuss it.


## Good practices

1. `git clone` the repo and create a new branch with `git checkout -b <new_branch>`.
2. Make your changes, test them, commit them, and push them to your branch.
3. You can open a pull request on GitHub when you're satisfied.

> **Note:** Use the following naming convention for new branches : new_branch = <github_username>/<task_tag>/<feature|bug>/<branch_description>

__Things don't need to be perfect for PRs to be opened, draft PRs are welcome__.

## Style guide for Python Code

Python code should follow the [PEP8 conventions](https://www.python.org/dev/peps/pep-0008/).

Autoformatters are programs that refactor your code to conform with PEP 8 automatically.
We use [`black`](https://github.com/psf/black), which autoformats code following most of the rules in PEP 8.
And we use [`pre-commit-hooks`](https://github.com/pre-commit/pre-commit-hooks) to automatically run black before `git commit`.

```bash
pre-commit install
```

Please be careful with the [naming conventions](https://peps.python.org/pep-0008/#naming-conventions), especially:
- Function and variable names should be `lowercase`, with words separated by underscores.
- Class names should be in `CapitalizedWords`.
- Constants should be `UPPERCASE`, with words separated by underscores.
- Choose a consistent naming for variables used in different scripts/functions.
- Comments should start with capital letters.

## Documentation
__All contributions to the source code should be documented__.

Docstrings follow the [Google format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html),
have a look at other docstrings in the codebase for examples.

## Unit tests
__All contributions to the source code should be unit-tested__.

### Asserts
- To test equality between torch tensors, use `torch.testing.assert_close`.

You can find additional resources on unit testing following the links below:

- https://www.educative.io/blog/unit-testing-best-practices-overview#unit-testing-best-practices (good practices)
- https://brightsec.com/blog/unit-testing-best-practices/#simple-tests (other good practices)
- https://emimartin.me/pytest_best_practices (specific for pytest)
