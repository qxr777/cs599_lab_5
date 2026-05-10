import matplotlib.pyplot as plt
from pathlib import Path

# 收集所有 cs599_*.png 图表
IMG_DIR = Path(__file__).parent
figures = sorted(IMG_DIR.glob("cs599_*.png"))

if not figures:
    print("⚠️ 未找到 cs599_*.png 文件，请先运行实验一至六")
    exit(1)

# 计算子图布局
n = len(figures)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes = axes.flatten() if hasattr(axes, '__iter__') else [axes]

for i, img_path in enumerate(figures):
    img = plt.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(img_path.name, fontsize=10, pad=8)
    axes[i].axis('off')

# 移除多余子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("CS599 Lab 5 - 实验结果汇总", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
output_path = IMG_DIR / "cs599_all_results_summary.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 汇总报告已保存: {output_path}")
print(f"   共包含 {n} 张图表: {', '.join(f.name for f in figures)}")
