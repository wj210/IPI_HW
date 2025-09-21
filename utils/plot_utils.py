from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _clamp01(x):
    return max(0.0, min(1.0, float(x)))

def _lerp(a, b, t):
    return a + (b - a) * t

def _rgb(r, g, b):
    return f"rgb({int(round(r))},{int(round(g))},{int(round(b))})"

def _seq_blue_to_red(v, vmin, vmax):
    """Sequential: vmin(blue) → vmax(red)."""
    if vmax == vmin:
        t = 0.5
    else:
        t = _clamp01((v - vmin) / (vmax - vmin))
    # blue (0,0,255) -> red (255,0,0)
    r = _lerp(0,   255, t)
    g = _lerp(255,   0, t)
    b = 0
    return _rgb(r, g, b)

def _div_bwr(v): 
    """Diverging Blue–White–Red for v in [-1, 1]."""
    v = max(-1.0, min(1.0, float(v)))
    if v >= 0:
        # white (255,255,255) -> red (255,0,0)
        t = v  # 0..1
        r = 255
        g = _lerp(255, 0, t)
        b = _lerp(255, 0, t)
    else:
        # blue (0,0,255) -> white (255,255,255) as v goes -1..0
        t = -v  # 0..1
        r = _lerp(0, 255, t)
        g = 255
        b = _lerp(0, 255, t)
    return _rgb(r, g, b)

def _separate_normalize(values):
    """Map raw values to [-1,1] by scaling positives by max_pos and negatives by |min_neg|."""
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    pos = values > 0
    neg = values < 0
    if np.any(pos):
        mpos = values[pos].max()
        if mpos > 0:
            out[pos] = values[pos] / mpos
    if np.any(neg):
        mneg = values[neg].min()  # most negative
        if mneg < 0:
            out[neg] = values[neg] / abs(mneg)
    return out  # in [-1,1]

def html_colored_tokens_with_colorbar(tokens, scores, num_ticks=6, normalize=False, width_px=420):
    scores = np.asarray(scores, dtype=float)

    if normalize:
        nscores = _separate_normalize(scores)  # in [-1,1]
        min_val, max_val = -1.0, 1.0
        ticks = np.linspace(min_val, max_val, num_ticks)
        # gradient: blue -> white -> red
        gradient = "linear-gradient(to right, rgb(0,255,0), rgb(255,0,0))"
        # color fn on normalized values
        color_fn = lambda s: _div_bwr(s)
        # label values are the normalized ticks
        tick_vals_for_labels = ticks
    else:
        min_val = float(np.min(scores))
        max_val = float(np.max(scores))
        if min_val == max_val:
            # fall back to symmetric tiny span to avoid /0 while keeping colors stable
            min_val -= 1e-9
            max_val += 1e-9
        ticks = np.linspace(min_val, max_val, num_ticks)
        # gradient: blue -> red
        gradient = "linear-gradient(to right, rgb(0,255,0), rgb(255,0,0))"
        color_fn = lambda s: _seq_blue_to_red(s, min_val, max_val)
        tick_vals_for_labels = ticks

    tick_labels = ''.join(
        f'<span style="display:inline-block;position:absolute;left:{i/(num_ticks-1)*100}%;'
        f'transform:translateX(-50%);font-size:11px;">{tick:.2f}</span>'
        for i, tick in enumerate(tick_vals_for_labels)
    )

    colorbar_html = f"""
    <div style="position: relative; width: {width_px}px; height: 22px; margin: 6px 0;
        background: {gradient};
        border-radius: 5px; border:1px solid #888;">
      <div style="position: absolute; width: 100%; top: 22px; left: 0;">{tick_labels}</div>
    </div>
    <div style="width: {width_px}px; text-align:center; margin-bottom:10px; font-size:12px; color:#333;">score</div>
    """

    if normalize:
        colored = [
            f'<span style="background:{color_fn(s)}; color:black; padding:2px 6px; margin:2px; border-radius:4px;">{t}</span>'
            for t, s in zip(tokens, nscores)
        ]
    else:
        colored = [
            f'<span style="background:{color_fn(s)}; color:black; padding:2px 6px; margin:2px; border-radius:4px;">{t}</span>'
            for t, s in zip(tokens, scores)
        ]

    html = colorbar_html + '<div style="margin-top:6px;">' + ' '.join(colored) + '</div>'
    display(HTML(html))

def plot_line(data,xlabel,ylabel,labels=[],title=None,xticks=None):
    if data.ndim == 1:
        data = data.reshape(1,-1)
    x = np.arange(data.shape[1])
    sns.set(style="darkgrid")
    for i,y in enumerate(data):
        if not len(labels):
            sns.lineplot(x=x, y=y, marker="o")
        else:
            sns.lineplot(x=x, y=y, marker="o",label = labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(labels):
        plt.legend()
    if xticks is not None:
        plt.xticks(x, xticks)
    plt.show()