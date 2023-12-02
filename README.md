# Football-Match-Event-Forecast-v2
Repo for the StatsBomb conference 2023 paper "An Events and 360 Data-driven Approach for Extracting Team Tactics and Evaluating Performance in Football" (C.Yeung & R. Bunker, 2023), similar StatsBomb open data (EURO 2020 and/or WCup 2022) can be used.
## Introduction
The collective behavior of opposing multi-agent teams has been extensively researched in game theory, robotics, and sports analytics. In sports, team tactics frequently encompass the individuals’ strategic spatial and eventual behavior and are denoted in sequences of events, which are also known as possession in football. Analysis of team tactics is critical for training, strategy, and, ultimately, team success. While conventional notation and statistical approaches provide valuable insights into team tactics, contextual information has been overlooked, and teams' performance was not evaluated. To consider the contextual information, we employed the sequential pattern mining algorithm PrefixSpan to extract team tactics from possession, the Neural Marked Spatio Temporal Point Process (NMSTPP) model to model expected team behavior for a fair comparison between teams, and the metrics Holistic Possession Utilization Score (HPUS) to evaluate possessions.  In experiments, We identified five team tactics, validated the NMSTPP when incorporated with StatsBomb 360 data, and analyzed English Premier League (EPL) teams in season 2022/2023, with the results visualized using radar plots and scatter plots with mean shift clustering. 
### NMSTPP+360 model
<p align="center">
  <img src="https://github.com/calvinyeungck/Football-Match-Event-Forecast-v2/blob/main/fig/NMSTPP%2B360%20model.png" style="width: 60%;">
</p>

### Radar plot
<p align="center">
  <img src="https://github.com/calvinyeungck/Football-Match-Event-Forecast-v2/blob/main/fig/radar_plot_avg.png" alt="alt text" style="width: 40%;" >
</p>

### Scatter plot and mean shift clustering
<p align="center">
  <img src="https://github.com/calvinyeungck/Football-Match-Event-Forecast-v2/blob/main/fig/Opponent%20Wing%20Based_cluster.png" alt="alt text" style="width: 50%;" >
</p>

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
python model.train_predict.py -param -ff
```
1b. Train the model and predict
```
python model.train_predict.py -ff
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
### Plot the radar plot and scatter plot with mean shift clustering
1. Radar plot
```
python analysis.radar_plot.py -wo
```
2. scatter plot
```
python analysis.clustering.py
```
## Reference
For technical details and full experimental results, please check [paper1](https://arxiv.org/abs/) and [paper2](https://arxiv.org/abs/2302.09276). Please consider citing our work if you find it helpful to yours:

```
@article{yeungevents,
  title={An Events and 360 Data-Driven Approach for Extracting Team Tactics and Evaluating Performance in Football},
  author={Yeung, Calvin and Bunker, Rory}
}

@article{yeung2023transformer,
  title={Transformer-Based Neural Marked Spatio Temporal Point Process Model for Football Match Events Analysis},
  author={Yeung, Calvin CK and Sit, Tony and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2302.09276},
  year={2023}
}
```
