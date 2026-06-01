# 🧠 Fundamentals of Neural Networks Course

## 📚 About

This repository contains practical exercises for the **Fundamentals of Neural Networks** course created by **Mike Romanov**.

## 🗂️ Program

1. Getting Started with PyTorch  
2. Gradient Descent Practice  
3. Backpropagation with NumPy  
4. Building Your First Neural Network in PyTorch  

## ⚙️ Requirements

This course is designed to run inside the **Laborantum** environment. To get started, make sure you have the following tools installed:

- [VS Code](https://code.visualstudio.com/) or any of its forks  
- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension for VS Code  
- [Laborantum](https://marketplace.visualstudio.com/items?itemName=Laborantum.laborantum) extension  
- [Docker](https://www.docker.com/) or [Podman](https://podman.io/)  
- Git (to clone the repository)  

## 🚀 Getting Started

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
cd your-repo
code .
```

Open the project folder in VS Code and allow it to open in a Dev Container when prompted.

## Working with the Course Tasks

### Save your current progress

Before updating course materials, always create a commit with your current work. This gives you a safe point to return to if an update changes more than expected.

```bash
git status
git add .laborantum/texts/course
git add .laborantum/src
git commit -m "Save current course progress"
```

If you have other files that you intentionally changed, add them too. If `git status` shows files you do not recognize, inspect them before committing.

### Set up the source repository

The source repository is the instructor's repository. In the examples below, replace:

- `<SOURCE-REPO-URL>` with the instructor repository URL
- `<YOUR-REPO-URL>` with your own repository URL

First, check how your repository is connected:

```bash
git remote -v
```

If `origin` points directly to the instructor's repository, your local repository was cloned from the source repository. If `origin` points to your own GitHub/GitLab account and the web page says it is forked from the instructor's repository, your repository is a fork.

#### Case 1: your repo was cloned from the source repository

If you cloned the instructor's repository directly, `origin` usually points to the source repository. In that case, keep the instructor repository as `source` and add your own repository as `origin` if you want to push your work.

```bash
git remote rename origin source
git remote add origin <YOUR-REPO-URL>
git remote -v
```

#### Case 2: your repo is a fork of the source repository

If your repository is a fork, `origin` should point to your fork. Add the instructor repository as `source`:

```bash
git remote add source <SOURCE-REPO-URL>
git remote -v
```

If `source` already exists, update it:

```bash
git remote set-url source <SOURCE-REPO-URL>
```

### Update your local repo with source changes

My recommended approach is to commit your work first, then merge from the source repository. This is gentler than rebasing for this course because Jupyter notebooks are large JSON files, and rebasing notebook-heavy histories can make conflicts harder to understand.

```bash
git status
git add .
git commit -m "Save work before course update"
git fetch source
git merge source/main
```

If the source branch is called `master` instead of `main`, use:

```bash
git merge source/master
```

After the merge, run the relevant notebook checks again.

### If there are conflicts

First, see which files are conflicted:

```bash
git status
```

For ordinary `.py`, `.md`, and `.yml` files, open the files, resolve the conflict markers, then commit:

```bash
git add <resolved-file>
git commit
```

For `.ipynb` conflicts, be careful. Notebooks are JSON files, so manual conflict resolution is easy to break. The safest practical options are:

- If the conflict is in a task you have not started, take the source version:

```bash
git checkout --theirs -- "<path-to-task.ipynb>"
git add "<path-to-task.ipynb>"
```

- If the conflict is in a task where you have important work, save a copy of your notebook first, take the source version, and then manually reapply your solution cells.

- If `nbdime` is available, use it to inspect notebook conflicts:

```bash
nbdiff-web "<path-to-task.ipynb>"
```

After resolving all conflicts:

```bash
git status
git commit
```

### Add only new tasks without updating existing ones

If you want to add only new tasks and leave existing tasks untouched, fetch the source repository and copy only the new task directories.

```bash
git fetch source
```

List task files available in the source branch:

```bash
git ls-tree -r --name-only source/main .laborantum/texts/course
```

Then check out only the new task directory you want:

```bash
git checkout source/main -- ".laborantum/texts/course/<chapter>/<new-task>"
git add ".laborantum/texts/course/<chapter>/<new-task>"
git commit -m "Add new course task"
```

This avoids changing notebooks you have already started.

### Update one task with a known bug fix

If a task has a known bug that was fixed in the source repository, update only that task directory:

```bash
git fetch source
git checkout source/main -- ".laborantum/texts/course/<chapter>/<task-name>"
git add ".laborantum/texts/course/<chapter>/<task-name>"
git commit -m "Update fixed course task"
```

This replaces the local version of that task with the fixed source version. If you already wrote solutions in that task, save your work before doing this and reapply your solution cells afterward.

## ✅ How to Check Your Work

The Laborantum extension includes an automated grading system that compares your solutions to the instructor's reference implementation. It runs locally inside Docker, so you don’t need internet access once everything is set up.

## 💪 Additional Resources

The course is designed to run fully on your local machine, guiding you step-by-step through building neural networks from scratch. You will be able to check whether the iterations run at all.

However, for getting the models training and achieving meaningful results, you may need access to additional compute resources. Consider using:

- Google Colab Pro
- Vast.ai
- Other cloud computing platforms
