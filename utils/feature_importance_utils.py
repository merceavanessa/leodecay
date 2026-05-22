import pandas as pd
import numpy as np
import shap

from src.utils.notation_utils import feature_to_physics_notation
from src.model_selection.whitelisted_kbest import WhitelistedSelectKBest

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LassoCV

def get_lasso_path_coefficients(x, y, model_config):
    tscv = TimeSeriesSplit(n_splits=model_config['n_splits'])
    whitelist_idx = x.columns.get_indexer(model_config['whitelisted_features'])

    model = make_pipeline(
        StandardScaler(),
        WhitelistedSelectKBest(
            k=model_config['k'],
            whitelist_idx=whitelist_idx,
            score_func=f_regression,
        ),
        LassoCV(
            cv=tscv,
            random_state=42,
            tol=model_config['tol'],
            alphas=model_config['alpha_numeric'],
            max_iter=model_config['max_iter'],
            n_jobs=-1
        )
    )
    target = y.columns[0]
    model.fit(x, y[target])
    coefs = model['lassocv'].path(model[:-1].transform(x), y, alphas=model_config['alpha_numeric'],
                                  max_iter=model_config['max_iter'])[1]
    coefs = coefs.reshape(coefs.shape[1], -1)
    feats_for_coefs = x.columns[model['whitelistedselectkbest'].get_support()]

    return model, coefs, feats_for_coefs

def compute_lasso_feature_importance(model, feature_names):
    coefficients = model.named_steps['lasso'].coef_
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef_signed': coefficients,
        'notation' : [feature_to_physics_notation(f) for f in feature_names]
    })
    coef_df['coef_abs'] = coef_df['coef_signed'].abs()
    coef_df['coef_normalized'] = coef_df['coef_abs'] / coef_df['coef_abs'].sum()
    coef_df['is_negative'] = coef_df['coef_signed'] < 0
    coef_df['coef_normalized_signed'] = coef_df['coef_normalized'] * coef_df['is_negative'].replace({True: -1, False: 1})
    return coef_df

def compute_shap_feature_importance(model, x_scaled, feature_names):
    explainer = shap.LinearExplainer(model['lasso'], x_scaled)
    shap_values = explainer(x_scaled)
    shap_mean_signed = shap_values.values.mean(axis=0)
    shap_mean_abs = np.abs(shap_values.values).mean(axis=0)

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_signed_shap': shap_mean_signed,
        'mean_abs_shap': shap_mean_abs
    })
    shap_total = shap_df['mean_abs_shap'].sum()
    shap_df['normalized_importance'] = shap_df['mean_abs_shap'] / shap_total
    shap_df['is_negative_shap'] = shap_df['mean_signed_shap'] < 0
    shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return shap_values, shap_df

def compute_combined_feature_importance(coef_df, shap_df):
    combined_df = pd.merge(coef_df, shap_df, on='feature', how='outer', suffixes=('_lasso', '_shap')).fillna(0)
    combined_df['combined_importance'] = (combined_df['coef_normalized'] + combined_df['normalized_importance']) / 2
    return combined_df

def compute_feature_importances(model, x):
    selected_features = x.columns.tolist()
    coef_df = compute_lasso_feature_importance(model, selected_features)

    x_scaled = model[:-1].transform(x)
    shap_values, shap_df = compute_shap_feature_importance(model, x_scaled, selected_features)
    combined_df = compute_combined_feature_importance(coef_df, shap_df)

    return coef_df, shap_df, combined_df, shap_values

def prepare_coefs_df(scores_and_coef_dfs):
    all_feats = []
    for sat, df_dict in scores_and_coef_dfs.items():
        coefs_with_shap = df_dict['no_val']['coefficients']['train']
        coefs = coefs_with_shap.melt(
            id_vars=['feature', 'notation'],
            value_vars=['normalized_importance', 'coef_normalized'],
            var_name='Type',
            value_name='Value'
        )
        coefs['Type'] = coefs['Type'].replace({
            'coef_normalized': 'LASSO',
            'normalized_importance': 'SHAP'
        })
        coefs['Satellite'] = sat
        all_feats.append(coefs)
    return pd.concat(all_feats, ignore_index=True)

def process_alt_feats(feats_all_sats, satellites, colors_by_feature, threshold=0.02, importance_kind='both'):
    feats = feats_all_sats[feats_all_sats['Satellite'].isin(satellites)].copy()
    if importance_kind == 'both':
        feats['Importance'] = feats.groupby(['feature', 'Type'])['Value'].transform('mean')
    elif importance_kind == 'lasso':
        feats = feats[feats['Type'] == 'LASSO']
        feats['Importance'] = feats['Value']
    elif importance_kind == 'shap':
        feats = feats[feats['Type'] == 'SHAP']
        feats['Importance'] = feats['Value']
    else:
        raise ValueError(f"Unknown importance_kind: {importance_kind}")

    feats.loc[feats['Importance'] < threshold, 'feature'] = 'Others'
    feats = feats.drop_duplicates(subset=['feature']).drop(columns=['Type', 'Value', 'Satellite'])
    feats['Importance'] = feats['Importance'] / feats['Importance'].sum()  # rescale to 0-1
    feats['Color'] = feats['feature'].apply(lambda x: colors_by_feature[x])
    feats['notation'] = feats['feature'].apply(feature_to_physics_notation)
    return feats.sort_values('Importance', ascending=False)