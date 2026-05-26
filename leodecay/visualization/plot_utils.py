import os
import numpy as np
import textwrap
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import re
from matplotlib.lines import Line2D

from ..utils.notation_utils import feature_to_physics_notation

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = np.array(mcolors.to_rgb(c))
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * c + amount * white)

def get_base_colors_for_alphabetical_sorted_features(feats):
    base_features = sorted(feats['feature'].apply(lambda x: re.sub(r'_\d+$', '', x)).unique())
    color_map = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20))
    ])

    base_colors = {bf: color_map[i] for i, bf in enumerate(base_features)}
    base_colors['Others'] = (0.5, 0.5, 0.5)
    return base_colors

def get_color_for_feature(feature, base_colors_dict):
    base_feature = re.sub(r'_\d+$', '', feature)
    return base_colors_dict.get(base_feature, (0.5, 0.5, 0.5))

def plot_lasso_importance_with_stacked_lagged_inputs_black_and_white(feats, satellite, filepath, color, label='', xlim=None):
    set_plot_aspect("pastel")
    fig_width, fig_height = 3.46, 4
    # fig_width, fig_height = 8, 4
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    feats['base_feature'] = feats['feature'].apply(lambda x: re.sub(r'_\d+$', '', x))
    feats['lag'] = feats['feature'].apply(
        lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else 0
    )
    feats = feats.sort_values(by=['base_feature', 'lag'])

    cum_importance = feats.groupby('base_feature')['Importance'].sum().sort_values(ascending=True)
    base_features_sorted = cum_importance.index.tolist()

    max_lag = feats['lag'].max() if feats['lag'].max() > 0 else 1
    bottoms = {bf: 0 for bf in base_features_sorted}
    for bf in base_features_sorted:
        for lag in sorted(feats['lag'].unique()):

            row = feats[(feats['base_feature'] == bf) & (feats['lag'] == lag)]
            if not row.empty:
                importance = row['Importance'].values[0]

                lighten_factor = lag / max_lag  # 0 lag = 0 (darkest), max lag = 1 (lightest)

                lag_color = lighten_color(color, amount=lighten_factor)

                plt.barh(bf, importance, left=bottoms[bf], color=lag_color,
                         edgecolor='black', linewidth=0.2)

                if importance > 0.02:  # adjust threshold for readability
                    plt.text(bottoms[bf] + importance / 2, bf, str(lag),
                             ha='center', va='center', fontsize=7, color='black')

                bottoms[bf] += importance

    if xlim:
         plt.xlim(0, xlim)

    plt.xlabel(f'Normalized Importance\n({label})')
    plt.title(f'Feature Importance for {satellite}')

    labels = [rf'${feature_to_physics_notation(bf)}$' for bf in base_features_sorted]
    plt.yticks(ticks=base_features_sorted, labels=labels)

    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Feature Lag",
        [lighten_color(color, 0), lighten_color(color, 1)]
    )

    norm = mpl.colors.Normalize(vmin=0, vmax=max_lag)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Feature Lag (minutes)")

    plt.subplots_adjust( bottom=0.15, right=0.9, left=0.15)

    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_lasso_importance_with_stacked_lagged_inputs(feats, satellite, filepath, base_colors, label='', width=50, xlim=None):
    set_plot_aspect("pastel")
    fig_width, fig_height = 3.46, 4

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    feats['base_feature'] = feats['feature'].apply(lambda x: re.sub(r'_\d+$', '', x))
    feats['lag'] = feats['feature'].apply(
        lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else 0
    )
    feats = feats.sort_values(by=['base_feature', 'lag'])

    cum_importance = feats.groupby('base_feature')['Importance'].sum().sort_values(ascending=True)
    base_features_sorted = cum_importance.index.tolist()

    max_lag = feats['lag'].max() if feats['lag'].max() > 0 else 1

    bottoms = {bf: 0 for bf in base_features_sorted}
    for bf in base_features_sorted:
        for lag in sorted(feats['lag'].unique()):

            row = feats[(feats['base_feature'] == bf) & (feats['lag'] == lag)]
            if not row.empty:
                importance = row['Importance'].values[0]

                lighten_factor = lag / max_lag  # 0 lag = 0 (darkest), max lag = 1 (lightest)
                lag_color = lighten_color(base_colors[bf], amount=lighten_factor)

                plt.barh(bf, importance, left=bottoms[bf], color=lag_color,
                         edgecolor='black', linewidth=0.2)

                if importance > 0.02:  # adjust threshold for readability
                    plt.text(bottoms[bf] + importance / 2, bf, str(lag),
                             ha='center', va='center', fontsize=7, color='black')

                bottoms[bf] += importance

    if xlim:
         plt.xlim(0, xlim)

    plt.xlabel(f'Normalized Importance\n({label})')
    plt.title(f'Feature Importance for {satellite}')

    labels = [rf'${feature_to_physics_notation(bf)}$' for bf in base_features_sorted]
    plt.yticks(ticks=base_features_sorted, labels=labels)

    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)

    handles_features = [plt.Rectangle((0,0),1,1, color=base_colors[bf], ec='black')
                        for bf in base_features_sorted]
    legend1 = plt.legend(handles_features, labels,
               title="Feature Color", bbox_to_anchor=(1, 0.2), loc='lower right')
    plt.gca().add_artist(legend1)

    dummy_base = (0,0,0)
    handles_lag = [
        mpatches.Rectangle((0,0), 1, 1, facecolor=lighten_color(dummy_base, 0), edgecolor='black'),
        mpatches.Rectangle((0,0), 1, 1, facecolor=lighten_color(dummy_base, 1), edgecolor='black')
    ]
    labels_lag = ["Dark - Present", f"Light - Past lags (minutes)"]

    plt.legend(handles_lag, labels_lag,
               title="Color Gradient", loc='lower right', fontsize=7)
    plt.subplots_adjust( bottom=0.15, right=0.95, left=0.15)

    plt.savefig(filepath, dpi=300)
    plt.close()

def set_plot_aspect(color_palette="pastel"):
    sns.set(style="whitegrid", font_scale=1.0)
    sns.set_context("paper", font_scale=1)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 10,
        'savefig.dpi': 600,
    })
    colors = sns.color_palette(color_palette)

    return colors

def plot_combined_importance(feats, satellite_list, filepath, label='', width=50):
    set_plot_aspect("pastel")
    plt.figure(figsize=(5, 8))

    y_pos = np.arange(len(feats))
    plt.barh(y_pos, feats['Importance'], color=feats['Color'])
    plt.yticks(y_pos, [f"${s}$" for s in feats['notation']])

    x_label = f"Normalized importance computed using {', '.join(satellite_list)} orbits\n{label}"
    x_label_wrapped = textwrap.fill(x_label, width=width, break_long_words=False)
    plt.xlabel(x_label_wrapped)

    plt.subplots_adjust(left=0.26, right=0.95, top=0.95, bottom=0.2)
    plt.tight_layout()

    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)

    plt.savefig(filepath)
    plt.close()

def plot_shap_summary(shap_values, x, feature_names, satellite, filepath, cmap='Pastel1'):
        set_plot_aspect(cmap)

        shap.summary_plot(
            shap_values, x, feature_names=feature_names, show=False
        )

        ax = plt.gca()
        ax.set_facecolor("white")
        ax.set_xlabel(
            f"SHAP Value (Impact on model output)\n{satellite}", color="black", fontdict={"fontsize": 30}
        )

        ax.set_ylabel("Feature", color="black", fontdict={"fontsize": 30})

        labels = [rf"${feature_to_physics_notation(label.get_text())}$" for label in ax.get_yticklabels()]
        # labels = [ label.replace('\\', '\\\\') for label in labels ]
        ax.set_yticklabels(labels, fontsize=25)

        ax.tick_params(axis="x", colors="black", labelsize=25)
        ax.tick_params(axis="y", colors="black", labelsize=25)

        fig = plt.gcf()
        cbar = fig.axes[-1]
        cbar.set_ylabel("Feature Value", rotation=90, fontsize=25, color="black")
        cbar.yaxis.set_tick_params(labelsize=25, colors="black")

        os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

# Helper functions
def add_cme_and_lead(ax, cme_durations_json, ymin, ymax, best_lag, bcolor='gray', alpha=0.5):
    for c, catalog in enumerate(cme_durations_json):
        text = 'ICME start in-situ (ICMECAT)' if 'Wind' in catalog else 'Geomagnetic storm start (R&C)' if 'RC' in catalog else 'IP Shock arrival (CCMC)'
        for k, event in enumerate(cme_durations_json[catalog]):
            cme_time = pd.to_datetime(event['arrival'])
            # catalog_acronym = catalog.split('_')[-1]
            ax.vlines(cme_time, ymin=ymin, ymax=ymax, color='red' if c == 0 else 'purple' if c == 1 else '#1f77b4',
                      linestyle='--', label=text if k==0 else None, alpha=alpha,  linewidth=0.75)

            if ('RC' in catalog) and (k == 2):
                ax.vlines(cme_time + pd.Timedelta(minutes=best_lag), ymin=ymin, ymax=ymax,
                      color=bcolor, linestyle='--', alpha=alpha,  linewidth=0.75)

                for j in range(int(ymin), int(ymax), int(ymax//10)):
                    ax.hlines(y=j, xmin=cme_time, xmax=cme_time + pd.Timedelta(minutes=best_lag),
                              color=bcolor, alpha=alpha)

def format_axes(ax, interval=2):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, alpha=0.5)

def format_axes_and_legends(axes, a2, train_index, best_lag, ymax, nrows, annotate_train_size=True, top_left=False):
    for k in range(nrows):
        if annotate_train_size:
            ymax_axis = ymax
            ymin_axis = 0
            if k == 0:
                ymax_axis = axes[k].get_ylim()[1]
                ymin_axis = axes[k].get_ylim()[0]

            axes[k].vlines(train_index[-1], ymin=0, ymax=ymax, color='gray', linestyle=':', label='Start of Test Period', alpha=0.5, linewidth=0.75)
            axes[k].vlines(train_index[-1] + pd.Timedelta(minutes=best_lag),  ymin=0, ymax=ymax,  color='gray',  linestyle='--', alpha=0.5, linewidth=0.75)

            axes[k].set_ylim(ymin_axis,  ymax_axis)

        format_axes(axes[k])

    axes[-1].set_xlabel("Time")

    # handles, labels = axes[0].get_legend_handles_labels()
    # handles2, labels2 = axes[nrows-1].get_legend_handles_labels()
    # handles.extend(handles2)

    handles, labels = axes[nrows-1].get_legend_handles_labels()
    # labels.extend(labels2)
    if a2:
        a2.legend().remove()

    axes[0].legend().remove()
    by_label = dict(zip(labels, handles))
    if top_left:
        axes[nrows-1].legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, facecolor='white', framealpha=0.5,
                             fontsize=8, ncol=2, columnspacing=0.5, handletextpad=0.5)
    else:
        axes[1].legend(by_label.values(), by_label.keys(), loc='lower left', bbox_to_anchor=(-0.05, -1), ncol=2, frameon=True, facecolor='white',
                       framealpha=0.5, fontsize=8, columnspacing=0.5, handletextpad=0.5)

def plot_results(io, cme_durations_json, col_to_plot=('|avg B|', '|avg B|'),
                 col2_to_plot=None,
                 best_lag=265, filepath='', annot='', cme_catalogs_to_plot=['L1_ICME_arrivals_RC']):
    set_plot_aspect("colorblind")
    fig_width, fig_height = 3.46, 4
    nrows = 2

    cme_durations_json = {k: v for k, v in cme_durations_json.items() if k in cme_catalogs_to_plot}

    train_data, test_data = io.read_all()

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=False)

    colors = [sns.color_palette("muted")[-1], sns.color_palette("Blues", n_colors=2)[1]]
    linestyle = '-'

    if col2_to_plot:
        a2 = axes[0].twinx()
    else:
        a2 = None

    for ax in axes:
        ax.set_xlim(train_data[0].index[0], test_data[0].index[-1])

    # Plot training features
    sns.lineplot(x=train_data[0].index, y=train_data[0][col_to_plot[0]], #label=col_to_plot[1],
                 color='black', alpha=0.3, linestyle=linestyle, ax=axes[0], linewidth=1)

    sns.lineplot(x=test_data[0].index, y=test_data[0][col_to_plot[0]],  # label=col_to_plot[1],
                 color='black', alpha=1, linestyle=linestyle, ax=axes[0], linewidth=1)

    axes[0].set_ylabel(col_to_plot[1])

    if col2_to_plot:
        sns.lineplot(x=train_data[0].index, y=train_data[0][col2_to_plot[0]],
                     color=colors[6], alpha=0.4, linestyle=linestyle, ax=a2)

    # Plot predictions and targets
    for df, label_suffix, alpha in zip([train_data, test_data],
                                       ['Train', 'Test'],
                                       [0.4, 1]):
        x_df, y_df, y_pred_df = df

        sns.lineplot(x=x_df.index + pd.Timedelta(minutes=best_lag),
                     y=y_df[io.target],
                     label=f'Decay Rate ({label_suffix} | Target)', #if label_suffix == 'Test' else None,
                     color=colors[0], alpha=alpha, linestyle=linestyle, ax=axes[nrows-1], linewidth=1)

        sns.lineplot(x=x_df.index + pd.Timedelta(minutes=best_lag),
                     y=y_pred_df['y_pred'],
                     label=f'Decay Rate ({label_suffix} | Prediction)', # if label_suffix == 'Test' else None,
                     color=colors[1], alpha=alpha, linestyle=linestyle, ax=axes[nrows-1], linewidth=1)

    if col2_to_plot:
        sns.lineplot(x=test_data[0].index, y=test_data[0][col2_to_plot[0]],
                     label=col2_to_plot[1], color=colors[6], alpha=0.8, linestyle=linestyle, ax=a2)

    axes[1].set_ylabel('Orbital Decay Rate ($\mathregular{m\ d^{-1}}$)')

    col_df_max = int(max(train_data[0][col_to_plot[0]].max(), test_data[0][col_to_plot[0]].max()))
    col_df_min = int(min(train_data[0][col_to_plot[0]].min(), test_data[0][col_to_plot[0]].min()))

    add_cme_and_lead(axes[0], cme_durations_json, col_df_min, col_df_max, best_lag)

    ymax = max(test_data[2]['y_pred'].max(), test_data[1][io.target].max()) + (test_data[1][io.target].max()//4)
    add_cme_and_lead(axes[nrows-1], cme_durations_json, 0, ymax, best_lag)

    ltime = train_data[0].index[-1]

    mid_time = ltime - pd.Timedelta(minutes=50)
                #pd.Timedelta(minutes=best_lag - 1) + \

    axes[nrows - 1].text(
        mid_time,
        ymax - (test_data[1][io.target].max()//4),
        f'|-| Prediction lead time\n of {best_lag} minutes', ha='left', fontsize=6,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

    format_axes_and_legends(axes, a2, train_data[0].index, best_lag, ymax, nrows)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.99, bottom=0.3, left=0.1, right=0.99)

    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_train_size_comparison(io_with_val, io_no_val, satellite, filepath):
    orig_colors = {
        'train': '#a6bddb',  # light blue
        'val': '#3690c0',  # medium blue
        'test': '#045a8d'  # dark blue
    }

    pred_colors = {
        'train': '#a8ddb5',  # light green
        'val': '#31a354',  # medium green
        'test': '#006d2c'  # dark green
    }

    set_plot_aspect("colorblind")
    fig_width, fig_height = 3.46, 4

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, score_type in enumerate(['with_val', 'no_val']):
        if score_type == 'no_val':
            (x_train, y_train, y_pred_train), (x_test, y_test, y_pred_test) = io_no_val.read_all()
        else:
            (x_train, y_train, y_pred_train), (x_val, y_val, y_pred_val), (x_test, y_test, y_pred_test) = io_with_val.read_all()

        axes[i].plot(y_train[io_no_val.target], label='Train True', color=orig_colors['train'], linestyle='--')
        axes[i].plot(y_pred_train['y_pred'], label='Train Pred', color=pred_colors['train'], linestyle='--')

        if score_type == 'with_val':
            axes[i].plot(y_val[io_no_val.target], label='Validation True', color=orig_colors['val'], linestyle=':')
            axes[i].plot(y_pred_val['y_pred'], label='Validation Pred', color=pred_colors['val'], linestyle=':')

        axes[i].plot(y_test[io_no_val.target], label='Test True', color=orig_colors['test'])
        axes[i].plot(y_pred_test['y_pred'], label='Test Pred', color=pred_colors['test'])

        axes[i].set_ylabel('Orbital Decay Rate ($\mathregular{m\ d^{-1}}$)')
        title_text = (f"Predictions for {satellite}.\nTraining set halved to include validation set."
                      if score_type == 'with_val'
                      else f"Using full training set.")
        axes[i].set_title(title_text)
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')

    fig.text(-0.05, 0.6, 'Orbital Decay Rate ($\mathregular{m\ d^{-1}}$)', va='center', rotation='vertical')

    plt.subplots_adjust(wspace=0.1, hspace=0.2, top=0.99, bottom=0.3, left=0.1, right=0.99)

    format_axes_and_legends(axes, None, x_train.index, best_lag=0, ymax=None, nrows=2, annotate_train_size=False, top_left=True)

    os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_lasso_path_with_shrinkage(model, coefs, feats_for_coefs, filepath=None, tol=1e-8):
    set_plot_aspect("PiYG")

    coefs_for_features = pd.DataFrame(coefs,
                                      index=feats_for_coefs,
                                      columns=model['lassocv'].alphas_)

    coefs_for_features['coef_signed'] = coefs_for_features[model['lassocv'].alpha_]
    coefs_for_features.sort_values('coef_signed', ascending=False, inplace=True)
    coefs_for_features.drop('coef_signed', axis=1, inplace=True)

    fig_width, fig_height = 3.46, 4
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    selected_colors = ['mediumvioletred' if coefs_for_features.loc[f][model['lassocv'].alpha_] != 0
                       else 'mediumseagreen'
                       for f in coefs_for_features.index]

    for i, feature in enumerate(coefs_for_features.index):
        selected = selected_colors[i] == 'mediumvioletred'
        label = f"{feature} {'- selected' if selected else ''}"
        ax.plot(model['lassocv'].alphas_,
                coefs_for_features.loc[feature],
                label=label,
                color=selected_colors[i])

    color_legend_handles = [
        Line2D([0], [0], color='mediumvioletred', lw=2, label='Selected Features'),
        Line2D([0], [0], color='mediumseagreen', lw=2, label='Not Selected Features')
    ]
    l1 = ax.legend(handles=color_legend_handles,
              loc='upper left',
              frameon=True,
              fontsize=6,
              title='Feature Selection',
              title_fontsize=7)

    ax.add_artist(l1)

    percent_shrinkage = (np.isclose(coefs, 0, atol=tol)).mean(axis=0)
    ax2 = ax.twinx()
    ax2.plot(model['lassocv'].alphas_, percent_shrinkage,
             color='black', linestyle=':',
             label='Percentage of Features with\nCoefficients Shrunk to 0')
    ax2.set_ylabel("Percentage of Features with\nCoefficients Shrunk to 0")
    ax2.set_ylim(0, 1)
    ax2.grid(False)

    ax.set_xlabel("Alpha - Regularization Strength (log scale)")
    ax.set_ylabel("Coefficient")
    ax.set_xscale("log")
    ax.set_xlim(model['lassocv'].alphas_.min(), model['lassocv'].alphas_.max())
    ax.hlines(0, model['lassocv'].alphas_.min(), model['lassocv'].alphas_.max(),
              color='orange', linestyle='-', label='Zero Coefficient Line')
    oa = ax.axvline(model['lassocv'].alpha_, color='red', linestyle='--',
                    label=f"Optimal Alpha: {model['lassocv'].alpha_:.3f}")
    ax.title.set_position([.5, 1.02])
    ax.grid(True, alpha=0.5)

    ax.legend(handles=[oa, ax2.get_lines()[0]],
              labels=[f"Optimal Alpha: {model['lassocv'].alpha_:.3f}",
                      "Percentage of features\nwith coefficients = 0"],
              loc='lower right', frameon=False, fontsize=6,
              bbox_to_anchor=(1.02, 0))

    fig.subplots_adjust(bottom=0.3, right=0.8, left=0.15, top=0.99)

    print(f"Optimal alpha: {model['lassocv'].alpha_:.3f}")

    if filepath:
        os.makedirs(filepath.rsplit('/', 1)[0], exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

