## Contributing

Work In Progress.


## Installation

```bash
# Create your own fork of the repository if required and replace whittle-org with your username
git clone git@github.com/whittle-org/whittle.git
cd whittle
pip install -e ".[dev]"  # Install what's here (the `.` part) and install the extra dev dependancies
```

Setup `pre-commit` to run on every commit

```bash
pre-commit install
```

## Testing

```bash
pytest
```

## Conventional commits and Commitizen

We use [commitizen](https://commitizen-tools.github.io/commitizen/) to manage commits.
This enforces conventional commits.

To make a commit, simply run:

```bash
cz commit
```

This will prompt you to enter a commit message and enforce conventional commit formatting.

If you do not use `cz commit` or make a commit with a conventional commit message, **your PR will not pass CI**.


## Signing commits

1. [Add a SSH (or GPG) key as a signing key to you GitHub account.](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#ssh-commit-signature-verification)
2. [Configure `git` to use the key.](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key#telling-git-about-your-ssh-key)
3. [Configure `commitizen` to use the key.](https://commitizen-tools.github.io/commitizen/config/#signing-commits)


```bash
cz commit
```

## Release

Update the version in `pyproject.toml` first, say to `X.Y.Z`.
If you maintain a changelog, update it.

This part just makes a versioned commit and tag for github and to be able to easily
find code at a specific version. It will also help with versioned documentation to have a tag.

```bash
git add pyproject.toml [changelog-file-if-any]
git commit -m "bump: X.Y.Z"
git tag X.Y.Z
git push --tags
git push
```

Then to release on PyPI:

```bash
pip install twine # If not already

rm -rf ./dist  # Remove anything currently occupying the dist folder
python -m build --sdist  # Build a source distribution
twine upload dist/*  # Publish to PyPI
```

## Documentation

View locally

```bash
mkdocs --serve
```

Build and deploy to GitHub Pages.
Make sure to specify the github tag you want to deploy.

```bash
mike deploy --push --update-aliases <TAG> "latest"
```
