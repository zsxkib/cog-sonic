# Sonic
Sonic: Shifting Focus to Global Audio Perception in Portrait Animation


<a href='https://jixiaozhong.github.io/Sonic/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href="http://demo.sonic.jixiaozhong.online/" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
<a href='https://arxiv.org/pdf/2411.16331'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <a href="https://huggingface.co/spaces/xiaozhongji/Sonic" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
    </a>


## Updates
**`2025/01/17`**: Our [**Online huggingface Demo**](https://huggingface.co/spaces/xiaozhongji/Sonic/) is released.

**`2025/01/14`**: Our inference code and weights are released. Stay tuned, we will continue to polish the model.

**`2024/12/16`**: Our [**Online Demo**](http://demo.sonic.jixiaozhong.online/) is released.

## Inference

### Installtion

- install pytorch
```shell
  pip3 install -r requirements.txt
```
- All models are stored in `checkpoints` by default, and the file structure is as follows
```shell
Sonic
  ├──checkpoints
  │  ├──Sonic
  │  │  ├──audio2bucket.pth
  │  │  ├──audio2token.pth
  │  │  ├──unet.pth
  │  ├──stable-video-diffusion-img2vid-xt
  │  │  ├──...
  │  ├──whisper-tiny
  │  │  ├──...
  │  ├──RIFE
  │  │  ├──flownet.pkl
  │  ├──yoloface_v5m.pt
  ├──...
```
Download by `huggingface-cli` follow
```shell
  python3 -m pip install "huggingface_hub[cli]"
  huggingface-cli download LeonJoe13/Sonic --local-dir  checkpoints
  huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir  checkpoints/stable-video-diffusion-img2vid-xt
  huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny
```

or manully download [pretrain model](https://drive.google.com/drive/folders/1oe8VTPUy0-MHHW2a_NJ1F8xL-0VN5G7W?usp=drive_link), [svd-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and [whisper-tiny](https://huggingface.co/openai/whisper-tiny) to checkpoints/ 


### Run demo
```shell
  python3 demo.py \
  '/path/to/input_image' \
  '/path/to/input_audio' \
  '/path/to/output_video'
```



 
## Citation
```
@misc{ji2024sonicshiftingfocusglobal,
      title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation}, 
      author={Xiaozhong Ji and Xiaobin Hu and Zhihong Xu and Junwei Zhu and Chuming Lin and Qingdong He and Jiangning Zhang and Donghao Luo and Yi Chen and Qin Lin and Qinglin Lu and Chengjie Wang},
      year={2024},
      eprint={2411.16331},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2411.16331}, 
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jixiaozhong/Sonic&type=Date)](https://star-history.com/#jixiaozhong/Sonic&Date)
