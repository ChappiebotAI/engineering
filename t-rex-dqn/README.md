## T-Rex Deep Q-Learning
This repo is written based on the report http://cs229.stanford.edu/proj2016/report/KeZhaoWei-AIForChromeOfflineDinosaurGame-report.pdf but including some changes.
We use websocket for capturing frames to learn, resize frames to 50x200 size and discard jumping frames (because t-rex cannot control when jumping).

![](http://www.huhmagazine.co.uk/images/uploaded/google_game_big.jpg)

Code base of t-rex-runner is taken from https://github.com/wayou/t-rex-runner. Some event listeners added for capturing game frames on this repo.

## Requirements
 - tensorflow 
 - pyramid
 - opencv
 - gevent == 1.0.2
 - gevent-socketio ( https://github.com/abourget/gevent-socketio )

## Run
 - For learning:
 	```bash
 	python capture.py --mode=learn --using_cuda=1
 	```
 - For playing:
 	```bash
 	python capture.py --mode=play --checkpoint=path_to_checkpoint
 	```

 - Open any browser type `localhost:1234`

## Trained network
 Trained models is save in the trained_models folder.
