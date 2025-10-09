import wandb
import pandas as pd
import json
# 登录你的账号（只需运行一次）
wandb.login()

# 使用公共 API
api = wandb.Api()

# 获取你的项目下的所有 run（替换下面的用户名和项目名）
runs = api.runs("1244300844-Peking University/huggingface")
print(runs)
ans = []
for run in runs:
    if run.name == '/root/chenxr/ms-swift/output/GRPO_LOOK_AGAIN/v15-20250421-152327':
        print(f"Run: {run.name}")
        history_df = run.history(samples=1000)
        print(history_df.columns.tolist())
        for step, mean_length, loss, reward, MultiModalAccuracyORM, Format in zip(history_df['train/global_step'], history_df['train/completions/mean_length'], history_df['train/loss'], history_df['train/reward'], history_df['train/rewards/MultiModalAccuracyORM/mean'], history_df['train/rewards/Format/mean']):
            if pd.notna(mean_length):
                print(step, mean_length, loss, reward, MultiModalAccuracyORM, Format)
                ans.append({'step': step, 'train/completions/mean_length': mean_length, 'train/loss':loss, 'train/reward':reward, 'train/rewards/MultiModalAccuracyORM/mean':MultiModalAccuracyORM, 'train/rewards/Format/mean':Format})

with open("mean_length.json", "w") as f:
    json.dump(ans, f, indent=2)
