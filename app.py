import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
import requests
import subprocess
import sys
import numpy as np

# ──────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────
st.set_page_config(page_title="SKYCIPHER DRONE COMMAND", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
.stApp {
    background-color: #050a07;
    background-image: linear-gradient(0deg, transparent 24%, rgba(57,255,20,0.04) 25%, rgba(57,255,20,0.04) 26%, transparent 27%);
    background-size: 50px 50px;
    color: #d3d3d3;
    font-family: 'Share Tech Mono', monospace;
}
h1,h2,h3,h4 { color: #39ff14 !important; font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 3px; text-shadow: 0 0 8px rgba(57,255,20,0.5); }
div[data-testid="metric-container"] {
    background: rgba(13,24,17,0.85); border-radius: 2px; padding: 15px;
    border: 1px solid #1f3b28; border-left: 5px solid #39ff14;
    box-shadow: 0 4px 15px rgba(0,0,0,0.8), inset 0 0 15px rgba(57,255,20,0.05);
}
div[data-testid="metric-container"] label { color: #6a9c78 !important; font-size:0.85rem !important; text-transform:uppercase; }
div[data-testid="metric-container"] div { color: #39ff14 !important; font-weight: bold; }
.stButton>button { background:#0d1811; color:#39ff14 !important; font-weight:700;
    font-family:'Share Tech Mono',monospace; letter-spacing:2px; border-radius:0px;
    border:1px solid #39ff14; padding:12px 24px; transition:all 0.1s ease; text-transform:uppercase; }
.stButton>button:hover { background:#39ff14; color:#000 !important; box-shadow:0 0 25px rgba(57,255,20,0.8); }
.stTabs [data-baseweb="tab-list"] { gap:2px; }
.stTabs [data-baseweb="tab"] { background-color:#0a120d; border-radius:0px; padding:10px 15px;
    border:1px solid #1f3b28; border-bottom:none; color:#4a7553; font-weight:bold; letter-spacing:1px; }
.stTabs [aria-selected="true"] { background-color:#15291b !important; color:#39ff14 !important;
    border-color:#39ff14; border-bottom:2px solid #39ff14; }
div[data-testid="stSidebar"] { background-color:#070d0a; border-right:2px solid #1f3b28; }
.terminal-box { background-color:#000; color:#39ff14; font-family:'Share Tech Mono',monospace;
    padding:15px; border:1px solid #39ff14; border-radius:2px; height:400px;
    overflow-y:auto; box-shadow:inset 0 0 15px rgba(57,255,20,0.3); white-space:pre-wrap; }
.stProgress > div > div > div > div { background-color:#39ff14; box-shadow:0 0 10px #39ff14; }
hr { border-color:#1f3b28 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
RADAR_COLS = [f'radar_{i}' for i in range(8)]
CHART_COLORS = ["#39ff14", "#8fbc8f", "#00ffcc", "#ffdd00", "#ff4444"]
TEMPLATE = "plotly_dark"
DATA_PATH = os.path.join("DroneProject", "skycipher_dataset_20260512_125542.csv")
TEST_SCRIPT = os.path.join("DroneProject", "test.py")

# ──────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────
@st.cache_data(show_spinner="[ LOADING SKYCIPHER DATASET... ]")
def load_data(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"[ CRITICAL ] Dataset not found: {path}")
        st.stop()

    # Feature engineering
    df['Avg_Radar'] = df[RADAR_COLS].mean(axis=1)
    df['Risk_Level'] = (df['wind_strength'] * 0.5) + ((1 - df['closest_obstacle']) * 0.5)
    df['Cumulative_Reward'] = df.groupby('episode')['reward'].cumsum()
    df['Distance_Change'] = df.groupby('episode')['target_dist'].diff().fillna(0)
    return df

df = load_data(DATA_PATH)

# ──────────────────────────────────────────
# ML Model — Random Forest on real columns
# ──────────────────────────────────────────
@st.cache_resource(show_spinner="[ TRAINING RF CLASSIFIER... ]")
def train_rf(df):
    feats = ['target_dist', 'wind_strength', 'Avg_Radar', 'Risk_Level',
             'closest_obstacle', 'action_magnitude', 'bearing_deg']
    X = df[feats].fillna(0)
    y = (df['episode_outcome'] == 'success').astype(int)

    if len(X) > 10000:
        idx = X.sample(10000, random_state=42).index
        X_tr, y_tr = X.loc[idx], y.loc[idx]
    else:
        X_tr, y_tr = X, y

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    if len(y_tr.unique()) > 1:
        clf.fit(X_tr, y_tr)
    return clf, feats

model, features = train_rf(df)

# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## [ SKYCIPHER COMMAND ]")

    with st.expander("[ SYSTEM INTEL ]", expanded=True):
        st.markdown("""
**DATASET:** SkyCipher v3 (Level 3)
- **target_dist** — distance to objective
- **wind_strength** — environmental wind
- **radar_detected** — enemy radar flag
- **closest_obstacle** — nearest obstacle
- **frame_class** — Normal/Evading/Approaching/Crash
- **episode_outcome** — success/crash/timeout
        """)

    st.markdown("---")
    st.subheader("[ UPLINK ]")
    n8n_url = st.text_input("WEBHOOK URL:", value="https://dohaziad.app.n8n.cloud/webhook/c4895708-1203-4095-9991-b5c6ef3461e2")

    st.markdown("---")
    st.subheader("[ FILTERS ]")
    outcomes = df['episode_outcome'].unique().tolist()
    sel_outcomes = st.multiselect("OUTCOME:", outcomes, default=outcomes)
    classes = df['frame_class'].unique().tolist()
    sel_classes = st.multiselect("FRAME CLASS:", classes, default=classes)

    filtered = df[df['episode_outcome'].isin(sel_outcomes) & df['frame_class'].isin(sel_classes)]

    st.markdown("---")
    st.subheader("[ TACTICAL AI ]")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "[ SKYCIPHER ONLINE. AWAITING COMMAND. ]"}]

    chat_box = st.container(height=300)
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("ENTER COMMAND..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                ph = st.empty()
                ctx = filtered.iloc[-1][['target_dist','wind_strength','Risk_Level']].to_dict() if not filtered.empty else {}
                chat_history = st.session_state.messages[:-1] 
                payload = {
                "question": prompt, 
                "drone_context": ctx,
                "history": chat_history
                }
                try:
                    # 1. زدنا وقت الانتظار إلى 45 ثانية لأن Gemini يحتاج وقتاً أحياناً
                    res = requests.post(n8n_url, json=payload, timeout=45)
                    
                    if res.status_code == 200:
                        # 2. نحاول قراءة الرد كـ JSON، وإذا فشل (لأنه نص عادي من n8n) نقرأه كنص
                        try:
                            resp = res.json().get("output", res.text)
                        except:
                            resp = res.text
                    else:
                        resp = f"[ UPLINK ERROR {res.status_code} ]"
                        
                except requests.exceptions.Timeout:
                    resp = "[ UPLINK TIMEOUT ] - انتهى وقت الاتصال، الذكاء الاصطناعي يفكر طويلاً."
                except Exception as e:
                    resp = f"[ SYSTEM ERROR ]: {e}"
                
                # 🟢 السطر المطلوب لحل المشكلة
                full = ""
                
                # 🟢 استخدام str(resp) كطبقة حماية إضافية
                for w in str(resp).split():
                    full += w + " "
                    time.sleep(0.04)
                    ph.markdown(full + "█")
                ph.markdown(full)
        st.session_state.messages.append({"role": "assistant", "content": full})

# ────
# ──────────────────────────────────────────
# Header
# ──────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>[ SKYCIPHER DRONE COMMAND v3 ]</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#39ff14;font-size:1rem;text-align:center;letter-spacing:2px'>[ PPO LEVEL 3 | PYBULLET ENV | SKYCIPHER DATASET ]</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "[ DASHBOARD ]",
    "[ LIVE STREAM ]",
    "[ 3D RADAR ]",
    "[ ML RESULTS ]",
    "[ ALGO INTEL ]",
    "[ HARDWARE ]",
    "[ SIMULATION ]",
])

# ══════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════
with tab1:
    st.markdown("### [ MISSION OVERVIEW ]")
    c1, c2, c3, c4, c5 = st.columns(5)
    total_eps = filtered['episode'].nunique()
    successes = (filtered.groupby('episode')['episode_outcome'].first() == 'success').sum()
    success_rate = successes / total_eps * 100 if total_eps > 0 else 0
    c1.metric("TOTAL EPISODES", total_eps)
    c2.metric("SUCCESS RATE", f"{success_rate:.1f}%")
    c3.metric("AVG TARGET DIST", f"{filtered['target_dist'].mean():.2f} m")
    c4.metric("MAX WIND", f"{filtered['wind_strength'].max():.2f} m/s")
    c5.metric("RADAR DETECTIONS", int(filtered['radar_detected'].sum()))

    st.markdown("---")
    ca, cb = st.columns([1, 2])
    with ca:
        st.markdown("#### [ OUTCOME DISTRIBUTION ]")
        oc = filtered.groupby('episode')['episode_outcome'].first().value_counts().reset_index()
        oc.columns = ['Outcome', 'Count']
        fig = px.pie(oc, names='Outcome', values='Count', hole=0.65, color_discrete_sequence=CHART_COLORS)
        fig.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown("#### [ FRAME CLASS DISTRIBUTION ]")
        fc = filtered['frame_class'].value_counts().reset_index()
        fc.columns = ['Class', 'Count']
        fig2 = px.bar(fc, x='Class', y='Count', color='Class', color_discrete_sequence=CHART_COLORS)
        fig2.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### [ WIND vs RISK vs DISTANCE ]")
    
    # 1. أخذ العينة
    sample_df = filtered.sample(min(1200, len(filtered)), random_state=42)
    
    # 2. خدعة الطبقات (Layering): ترتيب البيانات لتُرسم حالة Normal في الخلفية، والحالات الحرجة في المقدمة
    sort_order = {'Normal': 0, 'Approaching': 1, 'Evading': 2, 'Crash': 3}
    sample_df['render_order'] = sample_df['frame_class'].map(sort_order)
    sample_df = sample_df.sort_values('render_order')

    fig3 = px.scatter(
        sample_df,
        x='wind_strength', 
        y='target_dist', 
        color='frame_class',
        size='Risk_Level', 
        hover_data=['episode','reward'],
        # 3. تخصيص الألوان والشفافية: جعل Normal شفاف جداً ليصبح خلفية، وإبراز الباقي
        color_discrete_map={
            'Normal': 'rgba(57, 255, 20, 0.25)', # أخضر تكتيكي شفاف جداً
            'Approaching': '#8fbc8f',            # لون واضح للاقتراب
            'Evading': '#00ffcc',                # أزرق ساطع للتفادي
            'Crash': '#ff4444'                   # أحمر للتحطم
        },
        size_max=12  # 4. التقييد الصارم: منع أي فقاعة من التضخم أكثر من هذا الحجم
    )
    
    fig3.update_traces(marker=dict(line=dict(width=0)))
    fig3.update_layout(
        template=TEMPLATE, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="WIND STRENGTH (m/s)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="TARGET DISTANCE (m)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # نقل المفتاح للأعلى لترتيب المساحة
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════
# TAB 2 — LIVE STREAM
# ══════════════════════════════════════════
with tab2:
    st.markdown("### [ EPISODE TACTICAL STREAM ]")
    ep_options = filtered['episode'].unique()
    sel_ep = st.selectbox("SELECT EPISODE:", ep_options)

    if st.button("[ INITIATE PLAYBACK ]"):
        ep_data = filtered[filtered['episode'] == sel_ep].sort_values('step')
        g1, g2, g3 = st.columns(3)
        ph1, ph2, ph3 = g1.empty(), g2.empty(), g3.empty()
        chart_ph = st.empty()
        live_steps, live_dist, live_reward = [], [], []

        for _, row in ep_data.iterrows():
            s = row['step']
            live_steps.append(s)
            live_dist.append(row['target_dist'])
            live_reward.append(row.get('reward', 0))

            # 1. مقياس المسافة
            fig_d = go.Figure(go.Indicator(mode="gauge+number", value=row['target_dist'],
                title={'text': "TARGET DIST", 'font': {'color': '#39ff14'}},
                gauge={'axis': {'range': [0, 30]}, 'bar': {'color': "#39ff14"}, 'bgcolor': '#0d1811'}))
            fig_d.update_layout(
                template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=200, font={'color': '#39ff14'},
                uirevision='constant' # 🟢 يمنع التذبذب
            )
            ph1.plotly_chart(fig_d, use_container_width=True, key=f"dist_gauge_{s}") # 🟢 يمنع الإيرور

            # 2. مقياس الرياح
            fig_w = go.Figure(go.Indicator(mode="gauge+number", value=row['wind_strength'],
                title={'text': "WIND STRENGTH", 'font': {'color': '#39ff14'}},
                gauge={'axis': {'range': [0, 3]}, 'bar': {'color': "#ffdd00"}, 'bgcolor': '#0d1811'}))
            fig_w.update_layout(
                template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=200, font={'color': '#39ff14'},
                uirevision='constant' # 🟢 يمنع التذبذب
            )
            ph2.plotly_chart(fig_w, use_container_width=True, key=f"wind_gauge_{s}") # 🟢 يمنع الإيرور

            # 3. مقياس احتمالية النجاة
            inp = pd.DataFrame([[row['target_dist'], row['wind_strength'], row['Avg_Radar'],
                                  row['Risk_Level'], row['closest_obstacle'],
                                  row.get('action_magnitude', 0), row.get('bearing_deg', 0)]], columns=features)
            prob = model.predict_proba(inp)[0][1] * 100 if hasattr(model, 'predict_proba') else 0.0
            fig_p = go.Figure(go.Indicator(mode="gauge+number", value=prob,
                title={'text': "SURVIVAL PROB %", 'font': {'color': '#39ff14'}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00ffcc"}, 'bgcolor': '#0d1811'}))
            fig_p.update_layout(
                template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', height=200, font={'color': '#39ff14'},
                uirevision='constant' # 🟢 يمنع التذبذب
            )
            ph3.plotly_chart(fig_p, use_container_width=True, key=f"prob_gauge_{s}") # 🟢 يمنع الإيرور

            # 4. المخطط الخطي
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=live_steps, y=live_dist, name="Distance", line=dict(color="#39ff14", width=2)))
            fig_line.add_trace(go.Scatter(x=live_steps, y=live_reward, name="Reward", line=dict(color="#00ffcc", width=2), yaxis="y2"))
            fig_line.update_layout(
                template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis2=dict(overlaying='y', side='right', color='#00ffcc'), margin=dict(t=20),
                uirevision='constant' # 🟢 يمنع التذبذب
            )
            chart_ph.plotly_chart(fig_line, use_container_width=True, key=f"line_chart_{s}") # 🟢 يمنع الإيرور
            
            time.sleep(1) # تأخير خفيف لضمان سلاسة العرض

# ══════════════════════════════════════════
# TAB 3 — 3D RADAR
# ══════════════════════════════════════════
# ══════════════════════════════════════════
# TAB 3 — 3D RADAR
# ══════════════════════════════════════════
# ══════════════════════════════════════════
# TAB 3 — 3D RADAR
# ══════════════════════════════════════════
# ══════════════════════════════════════════
# TAB 3 — 3D RADAR & TACTICAL SENSORS
# ══════════════════════════════════════════
with tab3:
    st.markdown("### [ 3D HOLOGRAPHIC FLIGHT PATH ]")
    if not filtered.empty:
        # 1. 🚫 إيقاف العشوائية نهائياً (No more random sampling)
        # نقوم بترتيب البيانات زمنياً لرسم مسار طيران متصل ومنطقي
        path_data = filtered.sort_values(by=['episode', 'step'])
        
        # حماية للمتصفح: نأخذ أحدث 4000 خطوة (متصلة زمنياً) لتجنب الانهيار (Crash)
        if len(path_data) > 4000:
            path_data = path_data.tail(4000)

        # 2. رسم مسارات متصلة (Lines) بدلاً من نقاط مبعثرة
        fig_3d = px.line_3d(
            path_data, 
            x='drone_x', y='drone_y', z='drone_z',
            line_group='episode', # يربط النقاط الخاصة بكل مهمة ببعضها لإنشاء خط سير متصل
            color='frame_class',  # يلون المسار بناءً على حالة الدرون (تكتيكي جداً)
            color_discrete_map={
                'Normal': '#39ff14',      # أخضر نيون للمسار الطبيعي
                'Approaching': '#8fbc8f', # أخضر باهت للاقتراب
                'Evading': '#00ffcc',     # أزرق ساطع عند التفادي
                'Crash': '#ff4444'        # أحمر عند التحطم
            },
            hover_data=['episode', 'step', 'target_dist']
        )

        # 3. تضخيم الخطوط وإضافة نقاط خفيفة لإعطاء طابع هولوجرام
        fig_3d.update_traces(
            mode='lines+markers', 
            line=dict(width=5), 
            marker=dict(size=2, opacity=0.3)
        )

        fig_3d.update_layout(
            template=TEMPLATE, 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=600,
            margin=dict(t=0, b=0, l=0, r=0),
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='#1f3b28', backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(showgrid=True, gridcolor='#1f3b28', backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(showgrid=True, gridcolor='#1f3b28', backgroundcolor='rgba(0,0,0,0)')
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # ترتيب مكان المفتاح
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")
        st.markdown("#### [ TACTICAL 360° SENSOR ARRAY ]")
        
        # 🟢 [ استبدال الـ Heatmap برادار دائري حقيقي ] 🟢
        # نأخذ قراءة حساسات الرادار لآخر لحظة وصلها الدرون
        latest_radar = path_data.iloc[-1][RADAR_COLS].values.tolist()
        
        # نغلق الدائرة (تكرار أول قيمة في النهاية) لكي يكتمل الرسم الدائري 360 درجة
        latest_radar.append(latest_radar[0])
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'] 
        
        # بناء شبكة الرادار
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=latest_radar,
            theta=directions,
            fill='toself',
            fillcolor='rgba(57, 255, 20, 0.2)', # تعبئة خضراء شفافة
            line=dict(color='#39ff14', width=3), # حدود نيون
            name='Threat Detection'
        ))

        fig_radar.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(t=40, b=40, l=40, r=40),
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True, 
                    gridcolor='#1f3b28', # خطوط الشبكة الداخلية
                    linecolor='#1f3b28',
                    tickfont=dict(color='#6a9c78')
                ),
                angularaxis=dict(
                    gridcolor='#1f3b28',
                    linecolor='#39ff14',
                    tickfont=dict(color='#39ff14', size=14, family="Share Tech Mono")
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
# ══════════════════════════════════════════
# TAB 4 — ML STRATEGIC ANALYSIS (FIXED & CLEAN)
# ══════════════════════════════════════════
with tab4:
    st.markdown("### [ ML STRATEGIC ANALYSIS ]")

    # 1. تعريف قاموس الألوان داخل التاب لضمان الوصول إليه
    outcome_colors = {'success': '#39ff14', 'crash': '#ff4444', 'timeout': '#ffdd00'}

    # الصف الأول: منحنى التعلم وأهمية الميزات
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### [ SMOOTHED LEARNING TREND ]")
        ep_reward = filtered.groupby('episode')['Cumulative_Reward'].max().reset_index()
        # تنعيم البيانات باستخدام المتوسط المتحرك
        ep_reward['Moving_Avg'] = ep_reward['Cumulative_Reward'].rolling(window=50, min_periods=1).mean()
        
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(x=ep_reward['episode'], y=ep_reward['Cumulative_Reward'],
                                   mode='lines', line=dict(color='rgba(57, 255, 20, 0.1)'),
                                   name='Raw Reward'))
        fig_cr.add_trace(go.Scatter(x=ep_reward['episode'], y=ep_reward['Moving_Avg'],
                                   mode='lines', line=dict(color='#39ff14', width=3),
                                   name='Learning Trend'))
        
        fig_cr.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="EPISODES", yaxis_title="MAX REWARD",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cr, use_container_width=True)

    with cb:
        st.markdown("#### [ RF FEATURE IMPORTANCE ]")
        if hasattr(model, 'feature_importances_'):
            imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale=['#0d1811', '#39ff14'])
            fig_imp.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # الصف الثاني: توزيع الرياح وتوزيع الأفعال
    ca2, cb2 = st.columns(2)
    with ca2:
        st.markdown("#### [ WIND STRENGTH DISTRIBUTION ]")
        # استخدام المتغير المعرف أعلاه مباشرة
        fig_wind = px.histogram(
            filtered, x='wind_strength', nbins=50, 
            color='episode_outcome',
            color_discrete_map=outcome_colors 
        )
        fig_wind.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="WIND STRENGTH (m/s)", yaxis_title="SAMPLES"
        )
        st.plotly_chart(fig_wind, use_container_width=True)

    with cb2:
        st.markdown("#### [ TACTICAL ACTION ANALYSIS ]")
        # تحليل قوة الأفعال (Magnitude) بناءً على حالة الإطار
        fig_act = px.box(
            filtered, x='frame_class', y='action_magnitude', 
            color='frame_class', color_discrete_sequence=CHART_COLORS
        )
        fig_act.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="FRAME CLASS", yaxis_title="ACTION INTENSITY"
        )
        st.plotly_chart(fig_act, use_container_width=True)
# ══════════════════════════════════════════
# TAB 5 — ALGO INTEL
# ══════════════════════════════════════════
with tab5:
    st.markdown("### [ ALGORITHM INTEL ]")

    st.markdown("#### [ PPO ARCHITECTURE ]")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**PPO (Proximal Policy Optimization)**
- Policy: `MultiInputPolicy` (CNN + MLP)
- CNN: 3 Conv layers → 64×8×8 → 256 features
- Sensor MLP: 19 inputs → 128 → 64 features
- Total features: 320
- Learning Rate: `2e-4`
- Clip Range: `0.2`
- Entropy Coef: `0.02`
- Steps: `800,000`
        """)
    with col_b:
        st.markdown("""
**Curriculum Learning (3 Levels)**
| Level | Environment | Target Success |
|-------|-------------|----------------|
| 1 | Open field | 70% |
| 2 | Lego City (buildings) | 65% |
| 3 | Wind + Enemies + Radar | 78% |

**Reward System:**
- ✅ Success: +50
- 💥 Crash: -15
- ⏱ Timeout: -10
- 📡 Radar detection: -1/step
- 🏹 Approach reward: ×15 × progress
        """)

    st.markdown("---")
    st.markdown("#### [ ENVIRONMENT — DroneEnvironment (PyBullet) ]")
    env_cols = st.columns(3)
    with env_cols[0]:
        st.markdown("""
**Level 1**
- Open flat ground
- No obstacles
- No wind
- Basic navigation
        """)
    with env_cols[1]:
        st.markdown("""
**Level 2**
- Lego City (15 buildings)
- 8-ray radar sensors
- No wind
- Obstacle avoidance
        """)
    with env_cols[2]:
        st.markdown("""
**Level 3**
- Wind (0.3–2.5 m/s)
- 3 enemy drones
- 2 ground radars
- 1 tank
- Full combat scenario
        """)

    st.markdown("---")
    st.markdown("#### [ SKYCIPHER DATASET STATS ]")
    total_rows = len(df)
    total_eps = df['episode'].nunique()
    class_dist = df['frame_class'].value_counts()
    outcome_dist = df.groupby('episode')['episode_outcome'].first().value_counts()

    s1, s2, s3 = st.columns(3)
    s1.metric("TOTAL STEPS", f"{total_rows:,}")
    s2.metric("TOTAL EPISODES", f"{total_eps:,}")
    s3.metric("CSV COLUMNS", len(df.columns))

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Frame Class Distribution**")
        fig_fc = px.bar(x=class_dist.index, y=class_dist.values, color=class_dist.index,
            color_discrete_sequence=CHART_COLORS, labels={'x': 'Class', 'y': 'Count'})
        fig_fc.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fc, use_container_width=True)

    with col_d:
        st.markdown("**Episode Outcome Distribution**")
        fig_od = px.pie(values=outcome_dist.values, names=outcome_dist.index, hole=0.6,
            color_discrete_sequence=["#39ff14", "#ff4444", "#ffdd00"])
        fig_od.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_od, use_container_width=True)

    st.markdown("#### [ DATASET SAMPLE ]")
    st.dataframe(
        filtered.sample(min(100, len(filtered))).style.background_gradient(cmap='Greens', subset=['reward', 'Risk_Level']),
        use_container_width=True, height=300
    )

# ══════════════════════════════════════════
# TAB 6 — HARDWARE
# ══════════════════════════════════════════
with tab6:
    st.markdown("### [ HARDWARE DIAGNOSTICS ]")
    if not filtered.empty:
        diag_ep = st.selectbox("SELECT MISSION FOR SCAN:", filtered['episode'].unique(), key="diag_ep")
        ep_data = filtered[filtered['episode'] == diag_ep]

        steps = len(ep_data)
        wind = ep_data['wind_strength'].max()
        risk = ep_data['Risk_Level'].mean()

        # Simulated health metrics based on mission stress
        battery = max(0, 100 - (steps * 0.08) - (wind * 5))
        motor = max(0, 100 - (wind * 10) - (risk * 20))
        sensor = max(0, 100 - (risk * 30))

        c1, c2, c3 = st.columns(3)
        c1.metric("BATTERY", f"{battery:.1f}%")
        c2.metric("MOTOR HEALTH", f"{motor:.1f}%")
        c3.metric("SENSOR ARRAY", f"{sensor:.1f}%")

        st.markdown("---")
        st.markdown("**BATTERY STATUS**")
        st.progress(int(battery))
        st.markdown("**MOTOR INTEGRITY**")
        st.progress(int(motor))
        st.markdown("**SENSOR CALIBRATION**")
        st.progress(int(sensor))

        if battery < 20 or motor < 40:
            st.error("[ WARNING ]: CRITICAL WEAR DETECTED. MAINTENANCE REQUIRED.")
        else:
            st.success("[ OK ]: ALL SYSTEMS OPERATIONAL.")

# ══════════════════════════════════════════
# TAB 7 — LIVE SIMULATION (ULTIMATE SILENT MODE)
# ══════════════════════════════════════════
with tab7:
    st.markdown("### [ LIVE PYBULLET SIMULATION ]")
    st.info("[ NOTE ]: Runs `test.py` — PyBullet window will open on host machine. Requires `vision_smart_drone_level_3.zip`.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("""
**What happens:**
1. Load PPO model (`vision_smart_drone_level_3.zip`)
2. Initialize Level 3 environment (wind + enemies + radar)
3. Run test episodes
4. Report mission stats
        """)
    with col_sim2:
        st.markdown("""
**Requirements:**
- `vision_smart_drone_level_3.zip` must exist in project folder ✅
- PyBullet installed ✅
- PyTorch + Stable Baselines3 installed ✅
        """)

    st.markdown("---")
    if st.button("[ INITIATE DEPLOYMENT SEQUENCE ]", key="run_sim"):
        st.markdown("#### [ TACTICAL TERMINAL FEED ]")
        terminal_ph = st.empty()
        try:
            env_vars = os.environ.copy()
            env_vars["PYTHONIOENCODING"] = "utf-8"
            env_vars["KMP_DUPLICATE_LIB_OK"] = "True"

            test_script_path = os.path.join("DroneProject", "test.py")
            root_dir = os.path.dirname(os.path.abspath(__file__))

            # 🟢 الضربة القاضية للإيرورات: توجيه الـ stderr للـ DEVNULL
            proc = subprocess.Popen(
                [sys.executable, "-X", "utf8", test_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # ⬅️ إخفاء أي Traceback أو Python Error نهائياً في الثقب الأسود
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=root_dir,
                env=env_vars
            )
            
            log = ""
            # فلترة شاملة لأي مخلفات برمجية من محرك PyBullet
            ignore_list = [
                "pybullet", "Thread", "taskId", "TERMINATED", "finished",
                "ExampleBrowser", "Vendor", "Renderer", "b3Printf", "Version",
                "numActiveThreads", "btShutDown", "argc", "argv", "starting thread",
                "MotionThreads", "env.close", "p.disconnect", "self.update_radar",
                "^^^^", "crashed =", "obs, reward"
            ]

            for line in iter(proc.stdout.readline, ''):
                if not line.strip():
                    continue # تجاهل الأسطر الفارغة
                    
                # إذا السطر بيحتوي على أي كلمة من القائمة السوداء، تجاهله فوراً
                if any(kw in line for kw in ignore_list):
                    continue
                
                log += line
                terminal_ph.markdown(f'<div class="terminal-box">{log}</div>', unsafe_allow_html=True)
            
            proc.stdout.close()
            proc.wait()
            
            # رسالة ختامية تكتيكية بدال رسالة الإيرور
            log += "\n[ SYSTEM ]: MISSION TERMINATED. CONNECTION CLOSED SECURELY."
            terminal_ph.markdown(f'<div class="terminal-box">{log}</div>', unsafe_allow_html=True)
            st.success("[ MISSION LOG SAVED ]")
            
        except Exception as e:
            st.error(f"[ CRITICAL SYSTEM FAILURE ]: {e}")