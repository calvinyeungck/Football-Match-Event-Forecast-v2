# Football-Match-Event-Forecast-v2
Repo for the StatsBomb conference 2023
## Code
Create python env (Python 3.9.0)
```
conda create -n <environment-name> python=3.9.0 --file req.txt
```
### Create the required dataframe from StatsBomb JSON file
0. Joining frames to events using event_uuid in 360-frames and id in events
```
python dataset.json_to_df.py
```
1. Data preprocessing and feature creating as in the NMSTPP model
```
python dataset.preprocessing.py
```
2. Splitting the train/valid/test set
```
python dataset.train_valid_test_split.py
```
3. (Optional) Plotting the action, time, and zone
```
python dataset.plot.py
```
4. Train (Optional)  and Prediction
```
python model.prediction_final.py
```
5. Calculate the performance metrics
```
python analysis.metrics.py
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
