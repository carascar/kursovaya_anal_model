import matplotlib.pyplot as plt


def plot_values(x_values, y_values_dict, title="График", xlabel="Индекс", ylabel="Значение",
                show_markers=True, save_path=None):
    """
    Отрисовывает график по списку значений.

    :param values: список чисел
    :param title: заголовок графика
    :param xlabel: подпись оси X
    :param ylabel: подпись оси Y
    :param show_markers: показывать ли точки
    :param save_path: путь для сохранения (если None — только показать)
    """



    if not y_values_dict:
        raise ValueError("Список значений пуст")

    fig, ax = plt.subplots(figsize=(10, 6))

    marker = 'o' if show_markers else None
    for name, value in y_values_dict.items():
        ax.plot(x_values, value, marker=marker, linewidth=2, label=name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Сохранено в {save_path}")

    plt.tight_layout()
    plt.show()
