Contributors are welcome! To contribute please use the following steps.

1. Fork the **EasyFEA** Repository on GitHub via “Fork” on https://github.com/matnoel/EasyFEA.

2. On your machine, clone the repository and install EasyFEA in editable mode with:

    ```
    git clone https://github.com/YOURNAME/EasyFEA.git
    cd EasyFEA
    python -m pip install -e .
    ```

    In editable mode (`-e`), code completion functionality may be compromised in your integrated development environment ([IDE](https://fr.wikipedia.org/wiki/Environnement_de_d%C3%A9veloppement)). To enable code completion, it is necessary to include the **EasyFEA** directory path in the additional resolution paths for the import search.

    For example, in Visual Studio Code (VS Code), you can achieve this by adding `<folder>/EasyFEA/` to "Python > Analysis: **Extra Paths**" in the Pylance extension settings. Or you can add `<folder>/EasyFEA/EasyFEA/` to your PYTHONPATH. Here, `<folder>` indicates the path to the directory containing `EasyFEA/`.

3. To develop a **new feature**, start working in a new branch via the `git checkout -b my_new_feature` command.

4. Add new and changed files with ```git add ...``` and commit the changes with ```git commit -m "Update ..."```.

5. After implementing and validating your changes with **tests** (see `EasyFEA/tests/` for tips) you can push the branch to your fork for the first time with ```git push -u origin my_new_feature``` , then just use ```git push```.

6. Carry out some additional work, then repeat steps **4** and **5.**

7. Use the GitHub interface at https://github.com/YOURNAME/EasyFEA to create a pull-request for your changes.

**EasyFEA** is an emerging project with a strong commitment to growth and improvement. Your input and ideas are invaluable to me. I welcome your comments and advice with open arms, encouraging a culture of respect and kindness in our collaborative journey towards improvement.

# Update your copy of the main base of your patch

1. To update the **main** of your fork to the same state as the **main** of the **EasyFEA** repository:

    a. inform your local git repository once of the existence of the remote repository: ```git remote add upstream https://github.com/matnoel/EasyFEA```

    b. make sure you're on the right branch: ```git checkout main```

    c. **rebase** your current branch on top of the matnoel/EasyFEA **main**: ```git pull --rebase upstream main```

2. To update a branch with patches onto the updated **main** branch:
    
    a. ```git checkout my_new_feature```
    
    b. ```git rebase main```, this may generate rebase/merge-conflicts you should resolve now. If you get lost, you can always use ```git rebase --abort``` to abort the rebase attempt.
    
    c. after a rebase of a branch with commits which was already pushed to a remote, you have to force-push: ```git push --force```