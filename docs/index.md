## Viewing docs locally

```bash
mkdocs serve  # that's it
```

!!! todo "Deployment"

    I'll set up `mike` and document how to use that to deploy docs to github pages.

## Quickstart Mkdocs
This is your [bible](phttps://squidfunk.github.io/mkdocs-material/reference/)
This is just an overview of some features.

```python
print("hello world")
```

```bash
whittle --help  # (1)!
```

1. Look, an admonation! You can put `code` in here too!
```python
print("cool")
```

You can create tabbed blocks which are super useful.

=== "Python"

    ```python hl_lines="3-4"
    print("hello from tabbed block 1")

    print("Hello from highlight")
    print("Hello from highlight as well")
    ```

=== "cli"

    ```bash
    whittle --help
    ```

=== "Diagram using mermaid"

    ``` mermaid
    graph LR
      A[Start] --> B{Error?};
      B -->|Yes| C[Hmm...];
      C --> D[Debug];
      D --> B;
      B ---->|No| E[Yay!];
    ```

!!! warning

    This is a warning and set to be expanded with `!!!`

??? tip "My Tip name"

    This is a closed tip with `???`

To link to internal api docs, use the following.

```
[LS][whittle.search.local_search]
```

See [LS][whittle.search.local_search] for more information.

To link to api docs in other codebases, you can directly reference them
in much the same way you would import them, for example, this references
the [`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor] class in the `concurrent.futures` module.
The backticks are optional, they just make the link look like code.

```
[`ProcessPoolExecutor`][concurrent.futures.ProcessPoolExecutor]
```

You can use `()` like you normally would to link to html links,
i.e. [Python Docs](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor).

```
[Python Docs](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor).
```

You can also link directly to other pages in your repo if you need using a relative syntax,

```
[API](./contributing)
```

You can directly include the content of other files if you need, but I've rarely had a use for it. The main one
was really for including the `CONTIBUTING.md` file in the `docs/contributing.md` file.

This is all that has to be done. The path is relative to the root of the repo (where you execute the command from).
```
--8<-- "CONTRIBUTING.md"
```
