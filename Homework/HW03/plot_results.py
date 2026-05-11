import re
import matplotlib.pyplot as plt

def parse_log(filename):
    train_accs = []
    test_accs = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析类似于: =====> Epoch [24/25] Summary - Train Accuracy: 84.72%, Test Accuracy: 81.78% <=====
                match = re.search(r'Train Accuracy: ([\d.]+)%, Test Accuracy: ([\d.]+)%', line)
                if match:
                    train_accs.append(float(match.group(1)))
                    test_accs.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return train_accs, test_accs

# 读取四个日志
base_train, base_test = parse_log('training.log')
lr_train, lr_test = parse_log('training_lr.log')
struc_train, struc_test = parse_log('training_struc.log')
bs_train, bs_test = parse_log('training_bs.log')

# 绘图 - 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Train Accuracy
if base_train: ax1.plot(range(1, len(base_train)+1), base_train, marker='o', label='Baseline (bs=128, 3-CNN)')
if lr_train: ax1.plot(range(1, len(lr_train)+1), lr_train, marker='s', label='Exp A (Changed LR)')
if struc_train: ax1.plot(range(1, len(struc_train)+1), struc_train, marker='^', label='Exp B (2-CNN)')
if bs_train: ax1.plot(range(1, len(bs_train)+1), bs_train, marker='d', label='Exp C (bs=64)')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Accuracy (%)')
ax1.set_title('Train Accuracy Comparison on CIFAR-10')
ax1.legend()
ax1.grid(True)

# Plot Test Accuracy
if base_test: ax2.plot(range(1, len(base_test)+1), base_test, marker='o', label='Baseline (bs=128, 3-CNN)')
if lr_test: ax2.plot(range(1, len(lr_test)+1), lr_test, marker='s', label='Exp A (Changed LR)')
if struc_test: ax2.plot(range(1, len(struc_test)+1), struc_test, marker='^', label='Exp B (2-CNN)')
if bs_test: ax2.plot(range(1, len(bs_test)+1), bs_test, marker='d', label='Exp C (bs=64)')

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Test Accuracy Comparison on CIFAR-10')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
# 保存图片
plt.savefig('accuracy_comparison.png', dpi=300)
print("Plot saved to accuracy_comparison.png")
