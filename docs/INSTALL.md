# Installation

1. Clone repo and initialize submodules  
`git clone --recurse-submodules https://github.com/ecker-lab/SAM2-Animal-Tracking`  
2. Setup Environment (e.g. using conda, python>=3.10)
`conda create --name sam2animal python=3.12`  
`conda activate sam2animal`  
3. Install Pytorch (torch>=2.5.1, torchvision>=0.20.1)
`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126`
4. Install the repo  
`pip install -e .` 
5. Download SAM 2 checkpoints (instructions taken from sam2 repo)
```cd sam2/checkpoints && \
./download_ckpts.sh && \
cd ../..
```
6. (Optional) Setup separate MMDetection environment  
This is only recommended if there is  a specific reason to use a MMDetection detector.  
`conda create --name mm_detect python=3.12`  
`conda activate mm_detect`  
Then follow the [MMDetection installation instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html).  
