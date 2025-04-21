# Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation (InDiRec)


This is the Pytorch implementation for the paper: [SIGIR'25]Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation. (URL will be applied)

## Implementation
### Requirements

python>=3.9<br>
Pytorch >= 1.12.0 <br>
torchvision==0.13.0 <br>
torchaudio==0.12.0<br>
numpy==1.24.4 <br>
scipy==1.6.0 <br>
pandas==2.2.3<br>
faiss-gpu==1.7.2

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
