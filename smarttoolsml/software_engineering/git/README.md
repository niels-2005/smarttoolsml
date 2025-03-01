# Git Branching Guide

## Branches

### Create a new branch
```sh
git checkout -b new_branch
```

### Push code to new branch
```sh
git push origin new_branch
```

### Change to main branch and merge both
```sh
git checkout main
git merge new_branch
```

### Delete branch
```sh
git branch -d new_branch
```
