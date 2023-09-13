# BluNF: Blueprint Neural Field

By Robin Courant, Xi Wang, Marc Christie and Vicky Kalogeiton

ICCV-W [AI3DCC](https://ai3dcc.github.io/) 2023

*Note: This is the first version containing the training pipeline. We will update this repository with the editing pipeline and add the other datasets from the paper.*

## Installation

### Environment

Create a new conda environment:
```
conda create --name blunf -y python=3.7
conda activate blunf
```

Install a specific release of `nerfstudio`:
```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
git checkout 6136745d75b484bb2306fb99ef39d966daaf6df6
pip install --upgrade pip setuptools
pip install -e .
```
*Note: this project heavily relies on `nerfstudio` and follows its structure and logic.*


Install requirements:
```
pip install -r requirements.txt
```

### Dataset

Download the archive from [here](https://drive.google.com/file/d/1wBM9n3jFJNvPlh6ll7g7x-azZrXlIYeF/view?usp=drive_link):
```
gdown --id 1wBM9n3jFJNvPlh6ll7g7x-azZrXlIYeF
unzip data.zip -d ./
```

Each dataset comprises a `depth` folder with depth maps,  `rgb` folder with rgb views, `semantic_class` folder with semantic maps, `global_blueprint.pth` file with extracted ground truth blueprint, and `transforms.json` with camera poses of each view.

*Note: Currently, only replica room 0 is available; other datasets from the paper will be released soon.*

## Usage

To start training BluNF for room 0, run:
```
python src/train.py --config-name replica_room0 datamanager.data_dir=./data/ssr-replica/room_0
```
*Note: All outputs (including checkpoints) are stored in `./outputs` folder.*


To evaluate BluNF for room 0, run:
```
python src/eval.py --config-name replica_room0 datamanager.data_dir=./data/ssr-replica/room_0 trainer.load_dir=CHECKPOINT_PATH
```

To evaluate stereo (MVR) for room 0, run:
```
python src/eval.py --config-name replica_room0 datamanager.data_dir=./data/ssr-replica/room_0 model=stereo optim=stereo
```


## Citation

If you use it in your research, we would appreciate a citation via

```
@InProceedings{courant2023blunf,
    author    = {Courant, Robin and Wang, Xi and Christie, Marc and Kalogeiton, Vicky},
    title     = {BluNF: Blueprint Neural Field},
    booktitle = {Proceedings of the IEEE/CVF International on Conference of Computer Vision (ICCV) Workshops},
    year      = {2023},
}
```