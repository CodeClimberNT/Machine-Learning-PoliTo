import numpy as np
from matplotlib import pyplot as plt


class Visualizer:

    @staticmethod
    def plot_hist(
            data: np.ndarray,
            labels: np.ndarray,
            *,
            bins: int = 10,
            labels_name: dict[int, str] = None,
            title: str | None = None,
            x_label: str | None = None,
            y_label: str | None = None,
            show_grid: bool = False,
            invert_x_axis: bool = False,
            plot_each_feature: bool = False,
    ) -> None:
        colors, label_names, unique_labels = Visualizer.get_data_and_plot_info(labels, labels_name)

        if plot_each_feature:
            for i in range(data.shape[0]):
                plt.figure()
                for label, color in zip(unique_labels, colors):
                    data_by_label = data[i, labels == label]
                    plt.hist(
                        data_by_label,
                        color=color,
                        density=True,
                        bins=bins,
                        alpha=0.5,
                        label=label_names[label],
                    )
                feature_title = f"{title} - Feature {i + 1}" if title else f"Feature {i + 1}"
                plt.title(feature_title)
                if x_label:
                    plt.xlabel(x_label)
                if y_label:
                    plt.ylabel(y_label)
                plt.legend()
                plt.grid(show_grid)
                if invert_x_axis:
                    plt.gca().invert_xaxis()
                plt.show()
        else:
            plt.figure()
            for label, color in zip(unique_labels, colors):
                data_by_label = data[:, labels == label].flatten()
                plt.hist(
                    data_by_label,
                    color=color,
                    density=True,
                    bins=bins,
                    alpha=0.5,
                    label=label_names[label],
                )
            if title:
                plt.title(title)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            plt.legend()
            plt.grid(show_grid)
            if invert_x_axis:
                plt.gca().invert_xaxis()
            plt.show()

    @staticmethod
    def get_data_and_plot_info(labels, labels_name):
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        if labels_name and len(labels_name) == len(unique_labels):
            label_names = {label: labels_name[label] for label in unique_labels}
        else:
            label_names = {label: str(label) for label in unique_labels}
        return colors, label_names, unique_labels

    @staticmethod
    def plot_scatter_matrix(
            matrix: np.ndarray,
            labels: np.ndarray,
            labels_name: dict[int, str],
            *,
            title: str,
            x_label: str,
            y_label: str,
            grid: bool = True,
            invert_y_axis: bool = False,
            invert_x_axis: bool = False,
    ) -> None:

        plt.figure()

        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        if labels_name and len(labels_name) == len(unique_labels):
            unique_labels = labels_name

        for label, color in zip(unique_labels, colors):
            indices = np.where(labels == label)
            plt.scatter(
                matrix[0, indices],
                matrix[1, indices],
                color=color,
                label=labels_name[label],
            )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend()

        plt.grid(grid)

        if invert_y_axis:
            plt.gca().invert_yaxis()
        if invert_x_axis:
            plt.gca().invert_xaxis()

        plt.show()

    @staticmethod
    def project1_scatter_hist_pair_features(
            data: np.ndarray,
            labels: np.ndarray,
            *,
            bins: int = 10,
            alpha: float = 0.4,
            labels_name: dict[int, str] = None,
            title: str | None = None,
            custom_titles: dict[tuple[int, int], str] = None,
            custom_x_labels: dict[tuple[int, int], str] = None,
            custom_y_labels: dict[tuple[int, int], str] = None,
            show_legend: bool = False,
            show_grid: bool = False
    ) -> None:
        colors, label_names, unique_labels = Visualizer.get_data_and_plot_info(labels, labels_name)

        num_features = data.shape[0]
        pairs = [(i, i + 1) for i in range(0, num_features, 2)]
        for pair_index, (i, j) in enumerate(pairs):
            if j >= num_features:
                break  # Skip if there's no valid pair (i, j)

            fig, axs = plt.subplots(2, 2, figsize=(12, 12))

            # Diagonal histograms
            for ax_row in range(2):
                feature_index = i if ax_row == 0 else j
                ax = axs[ax_row, ax_row]
                for label, color in zip(unique_labels, colors):
                    data_by_label = data[feature_index, labels == label]
                    ax.hist(
                        data_by_label,
                        color=color,
                        density=True,
                        bins=bins,
                        alpha=alpha,
                        label=label_names[label],
                    )
                custom_title = custom_titles.get((feature_index, feature_index),
                                                 f"Histogram of Feature {feature_index + 1}") if custom_titles else f"Histogram of Feature {feature_index + 1}"
                custom_x_label = custom_x_labels.get((feature_index, feature_index),
                                                     "Value") if custom_x_labels else "Value"
                custom_y_label = custom_y_labels.get((feature_index, feature_index),
                                                     "Frequency") if custom_y_labels else "Frequency"

                ax.set_title(custom_title)
                ax.set_xlabel(custom_x_label)
                ax.set_ylabel(custom_y_label)
                ax.grid(show_grid)
                if show_legend:
                    # and ax_row == 0:
                    ax.legend()

            # Off-diagonal scatter plots
            for ax_row, ax_col in [(0, 1), (1, 0)]:
                ax = axs[ax_row, ax_col]
                x_feature = data[i, :] if ax_row == 0 else data[j, :]
                y_feature = data[j, :] if ax_col == 1 else data[i, :]
                for label, color in zip(unique_labels, colors):
                    ax.scatter(
                        x_feature[labels == label],
                        y_feature[labels == label],
                        color=color,
                        alpha=0.5,
                        label=label_names[label] if ax_row == 0 and ax_col == 1 else None,
                    )
                custom_title = custom_titles.get((i, j),
                                                 f"Scatter Plot of Feature {i + 1} vs Feature {j + 1}") if custom_titles else f"Scatter Plot of Feature {i + 1} vs Feature {j + 1}"
                custom_x_label = custom_x_labels.get((i, j),
                                                     f"Feature {i + 1}") if custom_x_labels else f"Feature {i + 1}"
                custom_y_label = custom_y_labels.get((i, j),
                                                     f"Feature {j + 1}") if custom_y_labels else f"Feature {j + 1}"

                ax.set_title(custom_title)
                ax.set_xlabel(custom_x_label)
                ax.set_ylabel(custom_y_label)
                ax.grid(show_grid)
                if show_legend and ax_row == 0 and ax_col == 1:
                    ax.legend()  # Add legend to scatter plot subplot

            # handles, labels_legend = ax.get_legend_handles_labels()
            # fig.legend(handles, labels_legend, loc='upper right')

            if title:
                fig.suptitle(title)

            plt.tight_layout(rect=(0, 0, 1, 0.95))
            plt.show()
