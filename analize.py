import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

RESULTS_DIR = Path("results")
ANALYSIS_DIR = RESULTS_DIR / "analysis_plots"
ANALYSIS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "benchmark_all_models.csv"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=0.8)


def parse_model_name(name):
    if "RTDETR" in name:
        if name.endswith("v2r50m"): return name[:-6], "v2r50m"
        if name.endswith("r50m"): return name[:-4], "r50m"
        if name.endswith("r18"): return name[:-3], "r18"
        if name.endswith("r34"): return name[:-3], "r34"
        if name.endswith("v4m"): return name[:-3], "v4m"
        if name.endswith("v4x"): return name[:-3], "v4x"
    return name[:-1], name[-1]


def get_size_category(size_str):
    if size_str in ['n', 't', 'r18']:
        return 'Nano_Tiny'
    elif size_str in ['s', 'r34']:
        return 'Small'
    elif size_str in ['m', 'r50m', 'v2r50m', 'v4m']:
        return 'Medium'
    elif size_str in ['l', 'c']:
        return 'Large'
    elif size_str in ['x', 'v4x']:
        return 'ExtraLarge'
    return 'Other'


def get_pareto_frontier(df, x_col, y_col):
    sorted_df = df.sort_values(by=x_col, ascending=False).reset_index(drop=True)
    pareto_front = [sorted_df.iloc[0]]
    max_y_so_far = sorted_df.iloc[0][y_col]
    for i in range(1, len(sorted_df)):
        if sorted_df.iloc[i][y_col] > max_y_so_far:
            pareto_front.append(sorted_df.iloc[i])
            max_y_so_far = sorted_df.iloc[i][y_col]
    return pd.DataFrame(pareto_front)


def plot_size_comparison(df, category_name):
    if df.empty: return
    df = df.sort_values(by='mAP50-95', ascending=False)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    bar_plot = sns.barplot(data=df, x='Модель', y='mAP50-95', alpha=0.8, color='steelblue', ax=ax1, label='mAP 50-95')
    ax1.set_ylabel('mAP 50-95', color='steelblue', fontweight='bold')
    ax1.set_ylim(df['mAP50-95'].min() * 0.8, df['mAP50-95'].max() * 1.1)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    line_plot = sns.lineplot(data=df, x='Модель', y='FPS', color='darkred', marker='o', linewidth=3, markersize=10,
                             ax=ax2, label='FPS')
    ax2.set_ylabel('Скорость (FPS)', color='darkred', fontweight='bold')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove() if ax1.get_legend() else None
    ax2.get_legend().remove() if ax2.get_legend() else None
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1, 1), frameon=True,
               facecolor='white')
    ax1.grid(False)
    ax2.grid(False)
    plt.title(f"Сравнение моделей: {category_name}", fontsize=16, pad=15)
    fig.tight_layout()
    plt.savefig(ANALYSIS_DIR / f"size_compare_{category_name}.png", dpi=300)
    plt.close()


def plot_map_by_object_size(df):
    """Научный график: сравнение точности топовых моделей на объектах разного размера"""
    top_models = df.loc[df.groupby('Family')['mAP50-95'].idxmax()].sort_values(by='mAP50-95', ascending=False)

    melted_df = pd.melt(top_models,
                        id_vars=['Модель'],
                        value_vars=['mAP small', 'mAP medium', 'mAP large'],
                        var_name='Размер объектов',
                        value_name='mAP')

    melted_df['Размер объектов'] = melted_df['Размер объектов'].map({
        'mAP small': 'Мелкие (AP_S)',
        'mAP medium': 'Средние (AP_M)',
        'mAP large': 'Крупные (AP_L)'
    })

    plt.figure(figsize=(14, 7))
    sns.barplot(data=melted_df, x='Модель', y='mAP', hue='Размер объектов', palette='viridis')

    plt.title("Эффективность детекции в зависимости от размера объектов (Top-1 модель каждого семейства)", fontsize=16,
              pad=15)
    plt.ylabel("Точность (mAP)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.legend(title="Тип объекта (по стандарту COCO)", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "4_map_by_object_size.png", dpi=300)
    plt.close()
    print("Сохранен научный график: 4_map_by_object_size.png")


def main():
    if not CSV_PATH.exists():
        print(f"Файл {CSV_PATH} не найден!")
        return

    df = pd.read_csv(CSV_PATH)
    df[['Family', 'Size_Code']] = df.apply(lambda row: pd.Series(parse_model_name(row['Модель'])), axis=1)
    df['Size_Category'] = df['Size_Code'].apply(get_size_category)
    df['FPS'] = pd.to_numeric(df['FPS'], errors='coerce')
    df = df.dropna(subset=['FPS'])

    plt.figure(figsize=(14, 9))
    sns.scatterplot(data=df, x='FPS', y='mAP50-95', hue='Family', style='Family', s=150, palette='tab10', alpha=0.8)
    pareto_df = get_pareto_frontier(df, 'FPS', 'mAP50-95')
    plt.plot(pareto_df['FPS'], pareto_df['mAP50-95'], color='red', linestyle='--', linewidth=2, label='Граница Парето',
             alpha=0.6)

    texts = []
    for i, row in df.iterrows():
        if row['Модель'] in pareto_df['Модель'].values or row['mAP50-95'] > 0.52 or row['FPS'] > 40:
            texts.append(plt.text(row['FPS'], row['mAP50-95'], row['Модель'], fontsize=9))
    if HAS_ADJUST_TEXT:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.title("Баланс Скорость vs Точность (Граница Парето)", fontsize=16, pad=15)
    plt.xlabel("Инференс (FPS) -> Выше лучше", fontsize=12)
    plt.ylabel("Точность (mAP 50-95) -> Выше лучше", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "1_pareto_frontier.png", dpi=300)
    plt.close()
    print("Сохранен график: 1_pareto_frontier.png")

    categories = df['Size_Category'].unique()
    for cat in categories:
        plot_size_comparison(df[df['Size_Category'] == cat], cat)
        print(f"Сохранен график для категории: {cat}")

    plot_map_by_object_size(df)


if __name__ == "__main__":
    main()