## Github Quick Guide
### Setup git
On Ubuntu:
```bash
$ sudo apt-get install git
```
On Mac Os:
```bash
$ brew install git
```

### Clone a repo
To clone a repo in your local machine. You can use one of two methods.

1. Use https link:
![](https://chappiebotai.github.io/images/github-quickguide/https-clone.png)
Https link, for example: https://github.com/ChappiebotAI/engineering.git

On terminal, do command line:
```bash
$ git clone https://github.com/ChappiebotAI/engineering.git
```

2. Use SSL link (recommend)
If you use `https` method, you must need password when commit, so you should use `SSL` method for ease of commiting. First, you have to setup public key for github. Please follow this [guide](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/) to create a SSH key. When you finish, your public key file was stored in `~/.ssh`. It is usually named `id_rsa.pub`. Continue, access github settings page, after clicking on  the `SSH and GPGs keys` section:
![](https://chappiebotai.github.io/images/github-quickguide/github-setting.png)

Create new SSH key for github by clicking on `New SSH key` button. At here, you need to copy all contents of the public key file (you can use command line to read all content a file `cat ~/.ssh/id_rsa.pub`) to `key` textbox.
![](https://chappiebotai.github.io/images/github-quickguide/github-ssh.png)


SSL link, for example: git@github.com:ChappiebotAI/engineering.git

On termianl, do command line:
```bash
$ git clone git@github.com:ChappiebotAI/engineering.git
```
Check you clone folder:
```bash
$ cd engineering
```
## Commit a change
Assume that you are in a git folder.
1. Change your some files.
2. Check status: `git status`
3. Add changes to commit:
    - All all changes (include create new files, update old files, delete files...): `git add -A`
    - Only update old file: `git add -u`
    - Only a file: `git add file_path`
4. Check status again before creating a commit: `git status`
5. Create a commit: `git commit -m "your message"`
6. Push to remote repo: `git push origin master`

## Update your local repo
When some changes occur on remote repo, you need to update your local repo before committing. Use:
```bash
$ git pull origin master
```

**Note**: `origin` is usually remote's name, `master` is usually remote's branch. You can see this by:
```bash
$ git remote -v
```

See more at: https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf
