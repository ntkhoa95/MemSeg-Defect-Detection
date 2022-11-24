# MemSeg-Defect-Detection
An unofficial implementation of [MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities](https://arxiv.org/pdf/2205.00908.pdf) using PyTorch.

### **TO USE**
1. Clone the repository
```
git clone https://github.com/ntkhoa95/MemSeg-Defect-Detection
cd MemSeg-Defect-Detection
```
2. Download two datasets
- Describable Textures Dataset (DTD)
```
https://www.robots.ox.ac.uk/~vgg/data/dtd/
```
- MVTec Dataset
```
https://www.mvtec.com/company/research/datasets/mvtec-ad
```
3. Run
```
python main.py --object_name capsule
```
4. Inference Mode
```
voila "inference.ipynb" --port 8866 --Voila.ip 127.0.0.1
```

### **RESULTS**
Using batch training of 8
*MVTec Dataset*

|    | target     |   AUROC-image |   AUROC-pixel |   AUPRO-pixel |
|---:|:-----------|--------------:|--------------:|--------------:|
|  0 | leather    |               |               |               |
|  1 | wood       |               |               |               |
|  2 | carpet     |               |               |               |
|  3 | capsule    |     97.89     |     98.48     |     95.69     |
|  4 | cable      |               |               |               |
|  5 | metal_nut  |               |               |               |
|  6 | tile       |               |               |               |
|  7 | grid       |               |               |               |
|  8 | bottle     |      100      |      98.59    |     95.10     |
|  9 | zipper     |               |               |               |
| 10 | transistor |               |               |               |
| 11 | hazelnut   |               |               |               |
| 12 | pill       |               |               |               |
|    | **Average**    |           |               |               |

### **CITATION**
```
@article{DBLP:journals/corr/abs-2205-00908,
  author    = {Minghui Yang and
               Peng Wu and
               Jing Liu and
               Hui Feng},
  title     = {MemSeg: {A} semi-supervised method for image surface defect detection
               using differences and commonalities},
  journal   = {CoRR},
  volume    = {abs/2205.00908},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.00908},
  doi       = {10.48550/arXiv.2205.00908},
  eprinttype = {arXiv},
  eprint    = {2205.00908},
  timestamp = {Tue, 03 May 2022 15:52:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-00908.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```