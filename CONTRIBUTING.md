# How to contribute?

1. `git clone` the repo and create a new branch with `git checkout -b <new_branch>`.
2. Make your changes, test them, commit them, and push them to your branch.
3. You can open a pull request on GitHub when you're satisfied.
4. Don't forget to delete the branch once the pull request has been merged.

> **Note:** Use the following naming convention for new branches : new_branch = <github_username>/<feature|bug>/<branch_description>

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
