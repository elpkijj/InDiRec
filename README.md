# Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation (InDiRec)


This is the Pytorch implementation for the paper: 

[[SIGIR'25] Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation](https://arxiv.org/pdf/2504.16077).

## Implementation
### Requirements
```
python>=3.9
Pytorch >= 1.12.0
torchvision==0.13.0
torchaudio==0.12.0
numpy==1.24.4
scipy==1.6.0
pandas==2.2.3
faiss-gpu==1.7.2
```
### Datasets
Five public datasets are included in `datasets` folder. (Beauty, Sports, Toys, Video, ML-1M)

### Evaluate InDiRec
Here are the trained models for the Beauty, Sports_and_Games, Toys_and_Games, Video, and ML-1M datasets, stored in the `./output` folder. <br>
You can evaluate these models directly on the test set by running the following command:

```
python main.py --data_name <Data_name> --model_idx 1 --eval_only
```
or you can evaluate all models using the following code:
```
bash scripts/eval_all.sh
```

### Train InDiRec
To train InDiRec on a specific dataset, you can run the following command: 
```
bash scripts/<Data_name>_run.sh
```
or you can train all models using the following code:
```
bash scripts/run_all.sh
```

The script will automatically train InDiRec, save the best model based on the validation set, and then evaluate it on the test set.

## Acknowledgment
This code is implemented based on [ICLRec](https://github.com/salesforce/ICLRec) and [DreamRec](https://github.com/YangZhengyi98/DreamRec). We thank the authors for providing efficient implementations.

## Citation
Please cite our work if it helps your research.
```
@misc{qu2025intentawarediffusioncontrastivelearning,
      title={Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation}, 
      author={Yuanpeng Qu and Hajime Nobuhara},
      year={2025},
      eprint={2504.16077},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.16077}, 
}
```
