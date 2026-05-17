# TiP4GENCode[Reconstruction Stage]

This repository contains the reconstruction stage of **TiP4GEN**.

The main entry point is `pipeline.py`. It takes a video or image folder, runs MonST3R pose estimation, then launches `train.py` and `render_c.py` to produce the reconstruction outputs.

## Setup

This project follows the environment style of [DreamScene360](https://github.com/ShijieZhou-UCLA/DreamScene360), but the text-to-pano pretrained models are **not needed** for this code path.

### 1. Main environment

```bash
git submodule update --init --recursive

conda create --name dreamscene360 python=3.8
conda activate dreamscene360

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
pip install py360convert imageio[ffmpeg] lpips timm
pip install -e submodules/simple-knn
pip install -e submodules/diff-gaussian-rasterization-depth
```

Also place the Omnidata checkpoint here:

```bash
mkdir -p pre_checkpoints

# Download Omnidata depth checkpoint.
# Thanks to PERF for providing the model:
# https://github.com/perf-project/PeRF/tree/master/pre_checkpoints
tmp_dir="$(mktemp -d)"
wget -O "$tmp_dir/pre_checkpoints.zip" "https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=1"
unzip -o "$tmp_dir/pre_checkpoints.zip" -d "$tmp_dir/pre_checkpoints" || test -f "$tmp_dir/pre_checkpoints/omnidata_dpt_depth_v2.ckpt"
find "$tmp_dir/pre_checkpoints" -name "omnidata_dpt_depth_v2.ckpt" -exec cp {} pre_checkpoints/ \;
test -f pre_checkpoints/omnidata_dpt_depth_v2.ckpt
rm -rf "$tmp_dir"
```

### 2. MonST3R

Clone and install [junyi42/monst3r](https://github.com/junyi42/monst3r) in a separate environment named `monst3r`:

```bash
git clone --recursive https://github.com/junyi42/monst3r
cd monst3r
conda create -n monst3r python=3.11 cmake=3.14.0
conda activate monst3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
cd data
bash download_ckpt.sh
```

Update the hardcoded MonST3R path in `pipeline.py` to match your machine.

## Usage

```bash
python pipeline.py --input_path <path_to_video_or_image_folder> --post train
```

`pipeline.py` will:

1. Convert the input into perspective views
2. Run MonST3R pose estimation
3. Train each frame
4. Render results and pack preview videos

## Output

Results are written under:

- `data/per/<parent>/<scene>/`
- `output/4d_exp/<parent>/<scene>/<frame>/`

## Notes

- The code currently assumes a Linux-style MonST3R path in `pipeline.py`.
- `train.py` and `render_c.py` can also be run separately if needed.

## Citation

```
@inproceedings{xing2025tip4gen,
  title={Tip4gen: Text to immersive panorama 4d scene generation},
  author={Xing, Ke and Liang, Hanwen and Xu, Dejia and Yin, Yuyang and Plataniotis, Konstantinos N and Zhao, Yao and Wei, Yunchao},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={9267--9276},
  year={2025}
}
```
