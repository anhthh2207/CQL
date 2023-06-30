# CQL
Conservative Q learning

### Set up env
```bash
conda env create -f environment.yml
```

### Download Dataset
```bash
cd data
pip install git+https://github.com/takuseno/d4rl-atari
python download_dataset.py
```
### Train online
```bash
python train_online.py
```

### Train offline
```bash
python train_offline.py
```

