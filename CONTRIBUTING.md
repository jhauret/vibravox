# Welcome to contributing to Vibravox

If you're reading this, you're probably interested in contributing to Vibravox. We're happy to have you here!

## How to get started?

This repo is meant to be a collaborative project on several tasks related to speech processing recorded by body-conducted microphones. Here is the list of currently supported tasks:

| Task Tag | Description          |
|----------|----------------------|
| `bwe`    | Bandwidth Extension  |
| `stp`    | Speech to Phoneme    |
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

## Documentation
__All contributions to the source code should be documented__.

Docstrings follow the [Google format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html),
have a look at other docstrings in the codebase for examples.
