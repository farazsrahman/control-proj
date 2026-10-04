___

I am loading the overleaf Pranam made for NimblDrift into this repo as a *subtree*. 

The way to handle changes are as follows

Editing the latex:
- edit and commit as normal
- commits from the main repo cached for next subtree push

Pushing the latex:
```shell
git subtree push --prefix=overleaf overleaf main
```
- This will push the main repos changes as commits to the subtree

Pulling the latex:
```shell
git subtree pull --prefix=overleaf overleaf main --squash
```
- This will pull all changes to the overleaf down as one *single* commit
- Merge conflicts may need to be resolved - best to do this before editing.


To do this all again:
```sh
git subtree add --prefix=overleaf <external-repo-URL> main --squash
```