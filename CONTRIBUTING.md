Contributors are welcome! To contribute, please follow these steps.

1. Fork the **EasyFEA** Repository on GitHub via “Fork” on
   https://github.com/matnoel/EasyFEA.

2. On your machine, clone the repository and install EasyFEA in editable mode with:

   ```bash
   git clone https://github.com/YOURNAME/EasyFEA.git
   cd EasyFEA
   python -m pip install -e ".[dev,lint,io,viz]"
   ```

   In editable mode (`-e`), code completion functionality may be compromised in your
   integrated development environment
   ([IDE](https://en.wikipedia.org/wiki/Integrated_development_environment)). To enable
   code completion, it is necessary to include the **EasyFEA** directory path in the
   additional resolution paths for the import search.

   For example, in Visual Studio Code (VS Code), you can achieve this by adding
   `<folder>/EasyFEA/` to "Python > Analysis: **Extra Paths**" in the Pylance extension
   settings. Or you can add `<folder>/EasyFEA/EasyFEA/` to your PYTHONPATH. Here,
   `<folder>` indicates the path to the directory containing `EasyFEA/`.

3. To develop a **new feature**, start working in a new branch via the
   `git checkout -b my_new_feature` command.

4. Format, then add new and changed files with `git add ...` and commit the changes with
   `git commit -m "Update ..."`. The **Lint** check fails on unformatted code:

   ```bash
   make format
   ```

5. After implementing and validating your changes with **tests** (see
   [Running tests](#running-tests) and [Writing tests](#writing-tests)) you can push the
   branch to your fork for the first time with `git push -u origin my_new_feature`, then
   just use `git push`.

6. Carry out some additional work, then repeat steps **4** and **5.**

7. Use the GitHub interface at https://github.com/YOURNAME/EasyFEA to create a
   pull-request for your changes.

**EasyFEA** is an emerging project with a strong commitment to growth and improvement.
Your input and ideas are invaluable to me. I welcome your comments and advice with open
arms, encouraging a culture of respect and kindness in our collaborative journey towards
improvement.

# Update your copy of the main base of your patch

1. To update the **main** of your fork to the same state as the **main** of the
   **EasyFEA** repository:

   - inform your local git repository once of the existence of the remote repository:
     `git remote add upstream https://github.com/matnoel/EasyFEA`
   - make sure you're on the right branch: `git checkout main`
   - **rebase** your current branch on top of the matnoel/EasyFEA **main**:
     `git pull --rebase upstream main`

2. To update a branch with patches onto the updated **main** branch:

   - `git checkout my_new_feature`
   - `git rebase main`, this may generate rebase/merge-conflicts you should resolve now.
     If you get lost, you can always use `git rebase --abort` to abort the rebase
     attempt.
   - after a rebase of a branch with commits which was already pushed to a remote, you
     have to force-push: `git push --force`

# Running tests

```bash
pytest                       # whole suite, in parallel
pytest -n0 tests/FEM/...     # one file, serially (e.g. under a debugger)
pytest --cov                 # with branch coverage; fails below `fail_under` (pyproject.toml)
make test-mpi                # the `@pytest.mark.mpi` tests, on 2 ranks
```

Plain `pytest` skips the MPI tests; CI runs them in a job of their own.

# Writing tests

## What gets a test

Every name a user can reach: what a package `__init__` imports, what a public submodule
it imports defines (e.g. `IO.Gmsh.Load_mesh`), and all their methods and properties,
`_single` ones included.

Everything else (`__private` names, helpers no package exports) is internal. It gets no
test of its own: it is tested through the names that use it.

## Rules

1. **One test file per source file, one prefix per name.** The test file mirrors the
   module that defines the name, and each test starts with the name it tests:

   ```python
   # EasyFEA/FEM/_group_elem.py  ->  tests/FEM/group_elem_test.py
   def test_GroupElem__Get_Mapping_inverts_a_distorted_element(): ...
   ```

   Use `test_<Name>_<behaviour>`, or `test_<Class>_<member>_<behaviour>` for a method or
   a property; drop the class's leading `_`, keep the member's. A method several classes
   inherit is tested once, in its defining module's file, looping over the subclasses
   that override it. Shared meshes go in `tests/conftest.py`.

2. **Loop with `subtests` over `parametrize`.** When the same check applies to several
   cases, write one test that loops over them. Each case still passes or fails on its
   own, so one failure never hides the others:

   ```python
   def test_Mesh_area_is_positive(subtests):
       for elemType in ElemType.Get_2D():
           with subtests.test(elemType=elemType):
               ...
   ```

   Reach for `parametrize` only when each case needs its own expensive setup.

   Looping over every element type costs time, so do it only where the result really
   depends on the element type:

   - `tests/FEM`, when the value comes from element-specific code;
   - `tests/IO` and `tests/Viz`;
   - the boundary-condition methods of the simulations;
   - `Beam`.

   Anywhere else, one element type is enough.

3. **Test through the public API.** A test sees EasyFEA the way a user does: it builds
   its objects and checks its results with the names listed above, never with internal
   ones. Assert on `simu.Get_K_C_M_F()` or `mesh.area`, not on a `__private` attribute;
   import from `EasyFEA.FEM`, not from `EasyFEA.FEM._mesh`. Then refactoring the
   internals never breaks a test. If a check seems to need an internal name, either that
   name deserves to be public, or another public result can be checked instead.

4. **Check exact values, without solving.** Compare to a result known in closed form (a
   bar in tension, a linear heat slab), or to the assembled system (`Get_K_C_M_F`). When
   a test needs a solution, set it with `_Set_solutions` rather than calling `Solve`.
   Solve only when the solve itself is under test (`Solve`, `SolverType`,
   `Solver_Set_*`, `DIC.Solve`, …), on at most 4 elements, or for a `Beam` closed form.
   Full simulations are covered by the examples, which the documentation build runs.

5. **Plots only in `tests/Viz/`.**
