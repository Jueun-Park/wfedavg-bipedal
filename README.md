# WFedAvg Bipedal

## Execution Sequence
### WFedAvg Weight Grid Search
```bash
python simple_base_agent_test.py  # 8m
python wfedavg_save_clients.py  # 15m
python wfedavg_load_and_fed.py  # 120m
```

### Train and Test RND Intrinsic Reward
```bash
python rnd_train.py  # 2m
python rnd_get_intrinsic_rewards.py  # 9m, 48m (?)
```
