# Detection of isolated sign language using the Mediapipe Holistic tool and their classification based on the transformer architecture.

---
### Collecting data:
1. Download repo. 
``` bash
git clone https://github.com/dxli94/WLASL.git
```
2. Download raw videos. 
``` bash
cd start_kit
python video_downloader.py
```
3. Extract video samples from raw videos. 
``` bash
python preprocess.py
```
4. You should expect to see video samples under directory videos/.

---
### Preprocessing:
``` bash
python .\preprocessing.py --root=$ROOT_PATH --max_gloss_numb=25 --n_threads=4
```

---
### Classification:

**classification.py** contains: creating dataset, normalization and building model