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
If you use `https` method, you must need password when commit, so you should use `SSL` method for ease of commiting. First, you have to setup public key for github. Please follow this [guide](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/) to create a SSH key. When you finish, your public key file was stored in `~/.ssh`. It is usually named `id_rsa.pub`. Next, access github settings page:
![](https://chappiebotai.github.io/images/github-quickguide/github-setting.png)

SSL link, for example: git@github.com:ChappiebotAI/engineering.git

On termianl, do command line:
```bash
$ git clone git@github.com:ChappiebotAI/engineering.git
```

