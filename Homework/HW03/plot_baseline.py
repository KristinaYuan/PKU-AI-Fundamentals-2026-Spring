import re
import matplotlib.pyplot as plt

filename = 'training.log'
train_accs = []
test_accs = []
losses = []

current_epoch_losses = []

try:
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配 Loss
            loss_match = re.search(r'Loss:\s*([\d.]+)', line)
            if 'Step' in line and loss_match:
                current_epoch_losses.append(float(loss_match.group(1)))
            
            # 匹配 Accuracy
            summary_match = re.search(r'Train Accuracy:\s*([\d.]+)%,\s*Test Accuracy:\s*([\d.]+)%', line)
            if summary_match:
                train_accs.append(float(summary_match.group(1)))
                test_accs.append(float(summary_match.group(2)))
                if current_epoch_losses:
                    losses.append(sum(current_epoch_losses) / len(current_epoch_losses))
                    current_epoch_losses = []
except FileNotFoundError:
    print(f"File {filename} not found.")

epochs = range(1, len(train_accs) + 1)

fig, ax1 = plt.subplots(figsize=(8, 5))

# 左侧 Y轴: Accuracy
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy (%)', color='tab:blue')
ax1.plot(epochs, train_accs, marker='o', color='tab:blue', label='Train Accuracy')
ax1.plot(epochs, test_accs, marker='s', color='tab:cyan', label='Test Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 右侧 Y轴: Loss
ax2 = ax1.twinx()
ax2.set_ylabel('Average Loss', color='tab:red')
ax2.plot(epochs, losses, marker='^', color='tab:red', linestyle='--', label='Train Loss')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.suptitle('Baseline Training Metrics (Accuracy & Loss) over Epochs')
fig.tight_layout()

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

ax1.grid(True, alpha=0.3)
plt.savefig('baseline_metrics.png', dpi=300)
print("Plot saved to baseline_metrics.png")
