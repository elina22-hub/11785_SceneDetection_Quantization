# Scene Change Detection and Quantization
Pytorch implementation of beta-VAE model on scene-change detection, and explore quantization methods to reduce memory-bandwith and improve inference latency.

Folder structure is as follows.

```bash
├── README.md 
├── experiment_results                                          # experient info and result
│   ├── BetaVAE_architecture.txt
│   ├── DL\ Final\ Report\ Results\ Summary.xlsx
│   └── scene-change.csv
├── generated_pictures                                          # experiment pic
│   ├── README.txt
│   ├── avg_acc+beta=0.25+ld=10
│   │   └── ...
│   ├── high_acc+beta=2+ld=10
│   │   └── ...
│   └── low_acc+beta=2+ld=5
│       └── ...
├── quantization                                                # quantization code
│   ├── BetaVAE_Quantization.ipynb
│   └── BetaVAE_Quantization_batchsize=1.ipynb
├── sample_data                                                 # sample data of 5 scenes
│   └── ...
└── training                                                    # model training code
    ├── best_model                                              # trained models of different hyper-paramters
    │   └── ...
    ├── config.yaml                                             # model and experiment configuration 
    ├── model                                                   # beta vae model
    │   └── beta_vae.py
    ├── run.ipynb                                               # main training file
    ├── scene_detection.py                                      # scene_detection class that supports kld calculation
    └── trainer.py                                              # trainer class
```
