# Football-Match-Event-Forecast-v2
Repo for the StatsBomb conference 2023
## Code
Create python env (Python 3.9.0)
```
conda create -n <environment-name> python=3.9.0 --file req.txt
```
### Create the required dataframe from StatsBomb JSON file
1. Joining frames to events using event_uuid in 360-frames and id in events
```
python dataset.json_to_df.py
```
2. Data preprocessing and feature creating as in the NMSTPP model
```
python dataset.preprocessing.py
```
3. Splitting the train/valid/test set (train_df2/valid_df2/test_df2) were used for the paper
```
python dataset.train_valid_test_split.py 
```
### Model training and predicting
1a. Prediction with pre-trained parameter
```
python model.train_predict.py --param
```
1b. Train the model and predict
```
python model.train_predict.py
```
### Extract and evaluate the team tactics
1. Extract team tactics in actions
```
python seq_mining.sequential_pattern_mining_action.py -a PrefixSpan -s 0.3
```
2. Extract team tactics in zones
```
python seq_mining.sequential_pattern_mining_zone.py -a PrefixSpan -s 0.3
```
3. Calculate the possession metrics (HPUS and HPUS+)
```
python analysis.metrics.py
```
4. Create the dataframe that consists of the possession tactics and metrics
```
python analysis.sequences_pattern.py
```



## Reference
For technical details and full experimental results, please check [paper1](https://arxiv.org/abs/) and [paper2](https://arxiv.org/abs/2302.09276). Please consider citing our work if you find it helpful to yours:

```
@article{

}

@article{yeung2023transformer,
  title={Transformer-Based Neural Marked Spatio Temporal Point Process Model for Football Match Events Analysis},
  author={Yeung, Calvin CK and Sit, Tony and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2302.09276},
  year={2023}
}
```
