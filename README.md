# Next-Action Prediction from User Interaction Logs

This project predicts user behavior from interaction logs. It includes:

- Next-action prediction (classification) using LSTM and Transformer models
- No-assistance F1 score prediction (regression) using Ridge, Random Forest, and LSTM

## How to run

git clone https://github.com/muhammedsinanp/Next-Action-Prediction-from-UI-Logs-Using-LLM-Assisted-Weakly-Supervised-Data-Generation.git

1. Generate synthetic data (optional)
```bash
python synthetic_generation.py
```

2. Train next-action model
```bash
python train_next_action.py
```

3. Train regression model 
```bash
python no_assistance_f1.py
```


## Author
Muhammed Sinan Pulukool  
M.Sc. AI & Robotics – Hochschule Hof  
GitHub: https://github.com/muhammedsinanp
