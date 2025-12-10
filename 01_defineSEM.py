from unicodedata import name
import pandas as pd
import numpy as np
import semopy
from semopy import calc_stats
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# 1. 설정 (Configuration)
COMBINED_FILE = r'all_full_sleep_data_0toNull_with_Cluster_v3.csv'
RANDOM_SEED = 42
COLS = ['temp', 'humi', 'illu', 'hb', 'rp', 'sleep_depth', 'time_hour']
NODE_COLOR = '#b2dfdb'       
NODE_BORDER = '#004d40'      
COLOR_POS = '#d32f2f'        
COLOR_NEG = '#1976d2'

# 2. 데이터 로드 및 전처리 (Data Load)
data = pd.read_csv(COMBINED_FILE)
df = data[COLS].dropna()
target_cols = ['temp', 'humi', 'illu', 'hb', 'rp', 'sleep_depth']
original_len = len(df)

# df = df.sample(n=10000, random_state=RANDOM_SEED).reset_index(drop=True)
df = df[(df['rp'] >= 10) & (df['rp'] <= 50)]

Q1 = df['hb'].quantile(0.25)
Q3 = df['hb'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['hb'] >= lower_bound) & (df['hb'] <= upper_bound)]
print(f"1차 필터링(rp, hb) 완료: {original_len} -> {len(df)} (삭제됨: {original_len - len(df)})")

plt.figure(figsize=(15, 10))
for i, col in enumerate(target_cols):
    plt.subplot(2, 3, i+1)
    plt.boxplot(df[col], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#a0c4ff', color='#1e3a8a'),
                medianprops=dict(color='#b00020'),
                whiskerprops=dict(color='#1e3a8a'),
                capprops=dict(color='#1e3a8a'),
                flierprops=dict(marker='o', markerfacecolor="#ff242b", markersize=5, markeredgecolor='none'))
    plt.title(f'Distribution of {col}')
plt.tight_layout()  
plt.show()

# 3. 모델 정의 (Model Definition)
def run_sem():
    desc = """
    # Measurement Model
    env =~ temp + humi
    phy =~ hb + rp
    
    # Structural Model
    phy ~ env
    sleep_depth ~ phy + illu
    
    # Covariance
    temp ~~ humi
    illu ~~ illu
    hb ~~ rp
    """
    model = semopy.Model(desc)
    model.fit(df)
    return model

# 4. 시각화 (Visualization)
def draw_sem_fixed_width(model):
    stats = model.inspect(std_est=True)
    
    G = nx.DiGraph()
    latent = {'env', 'phy'}
    observed = {'temp', 'humi', 'illu', 'hb', 'rp', 'sleep_depth'}
    
    show_error_vars = {'temp', 'humi', 'illu', 'hb', 'rp'}
    
    edge_props = {}
    error_nodes = []
    error_values = {}
    
    for _, row in stats.iterrows():
        lval, op, rval = row['lval'], row['op'], row['rval']
        est = row['Est. Std'] if 'Est. Std' in row else row['Estimate']
        
        if op in ['=~', '~']:
            src, dst = (lval, rval) if op == '=~' else (rval, lval)
            G.add_edge(src, dst)
            edge_props[(src, dst)] = {'weight': est, 'type': 'direct'}
            
        elif op == '~~':
            if lval != rval:
                G.add_edge(lval, rval)
                edge_props[(lval, rval)] = {'weight': est, 'type': 'cov'}
            else:
                if lval in show_error_vars:
                    err_node = f"e_{lval}"
                    error_nodes.append(err_node)
                    error_values[err_node] = f"{est:.2f}"
                    
                    G.add_edge(err_node, lval)
                    edge_props[(err_node, lval)] = {'weight': est, 'type': 'error'}

    pos = {
        'temp': (-2.5, 0.3),
        'humi': (-2.5, -0.3),
        'illu': (-2.5, -0.9),
        'env': (-1.0, 0),
        'phy': (1.0, 0),
        'hb': (0.5, -0.6),
        'rp': (1.5, -0.6),
        'sleep_depth': (3.0, 0)
    }

    pos['e_temp'] = (-2.9, 0.3)
    pos['e_humi'] = (-2.9, -0.3)
    pos['e_illu'] = (-2.9, -0.9)
    pos['e_hb'] = (0.5, -0.95)
    pos['e_rp'] = (1.5, -0.95)

    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    
    nx.draw_networkx_nodes(G, pos, nodelist=list(latent), node_shape='o', 
                           node_size=4500, node_color=NODE_COLOR, edgecolors=NODE_BORDER, linewidths=2)
    
    valid_obs = [n for n in observed if n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=valid_obs, node_shape='s', 
                           node_size=3500, node_color=NODE_COLOR, edgecolors=NODE_BORDER, linewidths=2)

    valid_errs = [n for n in error_nodes if n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=valid_errs, node_shape='o',
                           node_size=1300, node_color='#eeeeee', edgecolors='gray', linewidths=1)
    
    nx.draw_networkx_labels(G, pos, labels={n: error_values[n] for n in valid_errs}, 
                            font_size=10, font_color='black')

    for (u, v), props in edge_props.items():
        est = props['weight']
        edge_type = props['type']
        
        if edge_type == 'error':
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray', width=1.5,
                                   arrowstyle='-|>', connectionstyle="arc3,rad=0.0", ax=ax)
        else:
            style = 'dashed' if edge_type == 'cov' else 'solid'
            connection = "arc3,rad=0.2" if edge_type == 'cov' else "arc3,rad=0.0"
            arrow_style = '<|-|>' if edge_type == 'cov' else '-|>'
            
            abs_est = abs(est)
            base_c = COLOR_POS if est >= 0 else COLOR_NEG
            alpha_val = 0.3 + (min(abs_est, 1.0) * 0.7)
            color = to_rgba(base_c, alpha_val)
            
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color], width=3.0, style=style,
                                   arrowsize=25, arrowstyle=arrow_style, connectionstyle=connection,
                                   min_source_margin=35, min_target_margin=35, ax=ax)
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): f"{est:.2f}"},
                                         font_color=NODE_BORDER, font_weight='bold', font_size=11,
                                         label_pos=0.5, rotate=False,
                                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    main_labels = {n: n for n in G.nodes() if n not in error_nodes}
    nx.draw_networkx_labels(G, pos, labels=main_labels, font_size=12, font_weight='bold', font_color='black')

    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_POS, lw=4, label='Positive (+)'),
        plt.Line2D([0], [0], color=COLOR_NEG, lw=4, label='Negative (-)'),
        plt.Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Covariance'),
        plt.Line2D([0], [0], color='gray', lw=0, marker='o', markerfacecolor='#eeeeee', label='Error Value'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.title("Path Diagram", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 5. 실행 및 결과 출력 (Execution)
model = run_sem() 

stats = calc_stats(model)
print("\n" + "="*3 + " Model Fit " + "="*3)
target_metrics = ['DoF', 'RMSEA', 'GFI', 'AIC', 'BIC', 'CFI', 'Chi2', 'NFI', 'TLI', 'p-value', 'LogLik']
available_metrics = [m for m in target_metrics if m in stats.columns]
print(stats.T.loc[available_metrics])
draw_sem_fixed_width(model)
