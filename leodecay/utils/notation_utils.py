import re

from IPython.core.display import Math
from IPython.core.display_functions import display

# todo this function is pretty bad
def feature_to_physics_notation(feature_name, units = False, long=False):
    if "diff" in feature_name:
        base, lag = feature_name.replace('_diff', ''), None
        symbol = feature_to_physics_notation(base)
        return fr"\Delta {symbol}"
    if '_' in feature_name:
        base, lag = feature_name.rsplit('_', 1)
    else:
        base, lag = feature_name, None

    mapping = {
        '|avg B|': r'|B| (nT)' if units else r'|B|',
        'Flow Speed (km/s': r'$v$',
        'Flow pressure': r'P',
        'Temperature (K)': r'T',
        'Temperature': r'T',
        'AsyH': r'AsyH',
        'Vx Velocity': r'V$_x$',
        'Vy Velocity': r'V$_y$',
        'Vz Velocity': r'V$_z$',
        'F10.7 (LASP)': r'F$_{10.7}$',
        'F30 (LASP)': r'F$_{30}$',
        'ap (LASP)': r'a$_{\mathrm{p}}$',
        'Kp (LASP)': r'K$_{\mathrm{p}}$',
        'SymD (Omni)': r'SymD',
        'SymH (Omni)': r'SymH',
        'AsyD (Omni)': r'AsyD',
        'AsyH (Omni)': r'AsyH',
        'By GSE': r'B$_{\mathrm{y}}$',
        'Bx GSE': r'B$_{\mathrm{x}}$',
        'Bz GSE': r'B$_{\mathrm{z}}$',
        'By GSM': r'B$_{\mathrm{y}}$',
        'Bz GSM': r'B$_{\mathrm{z}}$',
        'Proton density (n/cc)': r'$\rho$',
        'Magnetosonic mach number': r'M$_{\mathrm{ms}}$',
        'Alfven mach number': r'M$_A$',
        'Plasma beta': r'$\beta$',
        'Electric Field (Mv/m)': r'E',
        'Percent Interpolated': r'pInterp',
        '# fine scale Plasma points': r'N$_{\mathrm{plasma}}$',
        '# fine scale IMF points' : r'N$_{\mathrm{IMF}}$',
        'RMS SD B vector (nT)' : r'RMS$_{\mathrm{B}}$',
        'RMS SD B scalar (nT)' : r'RMS$_{\mathrm{|B|}}$',
        "Timeshift (seconds)": r'$\Delta_{bow} t$',
        "Time between observations (seconds)": r'$\Delta_{obs} t$',
        "RMS Timeshift (seconds)": r'RMS$_{\Delta_{bow} t}$',
    }
    symbol = None
    for key in mapping:
        if base.startswith(key):
            symbol = mapping[key]
            break

    if symbol is None:
        symbol = base

    if lag is not None:
        # lag = str(int(lag)) # convert to minutes given that lag=1 means 30 seconds
        if '_' in symbol:
            symbol = re.sub(r'_(\w+)$', r'_{\1,t-' + lag + '}', symbol)
        else:
            symbol = fr"{symbol}_{{t-{lag}}}"

    return symbol

def format_lasso_equation(coef_df, intercept, terms_per_line=3):
    lines = []
    current_line = []

    sort_by_column = 'coef_normalized'

    coef_df = coef_df.sort_values(by=sort_by_column, ascending=False).reset_index(drop=True)
    feature_names = coef_df['notation'].values
    coefficients = coef_df['coef_signed'].values

    for coef, name in zip(coefficients, feature_names):
        if abs(coef) < 1e-6:
            continue

        sign = '+' if coef > 0 else '-'
        coef_abs = abs(coef)

        term = f"{sign} {coef_abs:.3g} {name}"

        current_line.append(term)
        if len(current_line) >= terms_per_line:
            lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    latex_lines = [f"\\hat{{y}}_t = {intercept:.2f} "] +[f"& {line}" for line in lines]
    equation = r"$$\begin{aligned}" + "\n" + "\\\\\n".join(latex_lines) + "\n" + r"\end{aligned}$$"

    return equation

def pretty_print_lasso_equation(coef_df, intercept, terms_per_line=3):
    equation = format_lasso_equation(coef_df, intercept, terms_per_line)
    display(Math(equation))
    return equation
