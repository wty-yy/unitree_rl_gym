## Train
```bash
python legged_gym/scripts/train.py --task=go2 --num_envs 4096 --headless
python legged_gym/scripts/train.py --task=go2 --num_envs 8  # DEBUG
```
## Play
```bash
python legged_gym/scripts/play.py --task=go2 --num_envs 8  # load latest
python legged_gym/scripts/play.py --task=go2 --num_envs 8 --load_run Nov13_00-00-05_  # load specified run
```