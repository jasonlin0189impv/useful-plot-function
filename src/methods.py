from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BasicPlot:
    def __init__(self, df: pd.DataFrame, font_scale: float = 0.5):
        self.df = df
        self._set_plot_style(font_scale=font_scale)

    def _set_plot_style(self, font_scale) -> None:
        sns.set(style="darkgrid")
        sns.set_context(font_scale=font_scale, rc={"grid.linewidth": 0.6})

    def plot_distribution(self, data_col: str, label_col: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        sns.set(rc={"figure.figsize": figsize})
        sns.kdeplot(data=self.df, x=data_col, hue=label_col, cut=0, common_norm=False, alpha=1)

    def plot_boxplot(self, data_col: str, label_col: str, figsize: Tuple[int, int] = (4, 6)) -> None:
        sns.set(rc={"figure.figsize": figsize})
        sns.boxplot(data=self.df, x=data_col, y=label_col, palette="Set2")

    def plot_stack_barplot(self, data_col: str, label_col: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        cross_tab_prop = pd.crosstab(index=self.df[label_col], columns=self.df[data_col], normalize="index")
        cross_tab = pd.crosstab(index=self.df[label_col], columns=self.df[data_col])

        cross_tab_prop.plot(kind="bar", stacked=True, colormap="Pastel2", figsize=figsize)

        plt.legend(loc="lower left", ncol=1)
        plt.ylabel("Proportion")

        for n, x in enumerate([*cross_tab.index.values]):
            for m, (proportion, y_loc) in enumerate(zip(cross_tab_prop.loc[x], cross_tab_prop.loc[x].cumsum())):
                if m == 0:
                    plt.text(
                        x=n - 0.17,
                        y=y_loc,
                        s=f"{np.round(proportion * 100, 1)}%",
                        color="black",
                        fontsize=15,
                        fontweight="bold",
                    )

        plt.xticks(rotation=45, ha="right")
        plt.show()

    def plot_cluster_barplot(self, data_col: str, label_col: str, figsize: Tuple[int, int] = (10, 6), **kargs) -> None:
        tmp = self.df[[label_col, data_col]]
        binned_size = kargs.get("binned_size", 5)
        binned_label = kargs.get("binned_label", None)
        binned_interval = kargs.get("binned_interval", None)

        result_df = self._binning_data(
            df=tmp,
            data_col=data_col,
            label_col=label_col,
            binned_size=binned_size,
            binned_interval=binned_interval,
            binned_label=binned_label,
        )

        g = sns.catplot(
            x=f"{data_col}_binned",
            y=f"{data_col}_percent",
            hue=label_col,
            kind="bar",
            data=result_df,
            height=figsize[1],
            aspect=figsize[0] / figsize[1],
        )
        g.ax.set_ylim(0, 100)

    def _binning_data(
        self,
        df: pd.DataFrame,
        data_col: str,
        label_col: str,
        binned_size: int = 5,
        binned_interval: Optional[List[int]] = None,
        binned_label: Optional[List[str]] = None,
    ) -> None:
        """binning data

        If given user define binned, use pd.cut.
        Else, use qcut.
        """
        tmp_df = df.copy()
        if binned_interval is not None:
            tmp_df[f"{data_col}_binned"] = pd.cut(tmp_df[data_col], bins=binned_interval, labels=binned_label)

        else:
            tmp_df[f"{data_col}_binned"] = pd.qcut(
                tmp_df[data_col], q=binned_size, duplicates="drop", labels=binned_label
            )

        result_df = tmp_df.groupby([f"{data_col}_binned", label_col], as_index=False).count()
        sum_ = tmp_df.groupby(label_col).count()

        for label in tmp_df[label_col].unique():
            result_df.loc[result_df[label_col] == label, f"{data_col}_percent"] = (
                result_df[data_col] / sum_[data_col][label]
            )

        result_df[f"{data_col}_percent"] = result_df[f"{data_col}_percent"].mul(100)

        return result_df
