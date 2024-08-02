## Contributing

Work In Progress.


## Installation

```bash
# Create your own fork of the repository if required and replace whittle-org with your username
git clone git@github.com/whittle-org/whittle.git
cd whittle
pip install -e ".[dev]"  # Install what's here (the `.` part) and install the extra dev dependancies
```

## Testing

```bash
pytest
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
