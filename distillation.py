import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ==========================================
# 1. 페이지 설정 및 CSS
# ==========================================
st.set_page_config(page_title="Knowledge Distillation Simulator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    body { font-family: 'JetBrains Mono', monospace; }
    .info-box {
        background-color: #1e1e1e; padding: 20px; border-radius: 12px;
        border-left: 6px solid #9467bd; margin-top: 10px; margin-bottom: 20px;
    }
    .metric-container { display: flex; justify-content: space-around; background: #2c2f33; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 공통 함수
# ==========================================
def softmax_with_temp(logits, T):
    # 수치적 안정성을 위해 max 값을 뺌
    exp_logits = np.exp((logits - np.max(logits)) / T)
    return exp_logits / np.sum(exp_logits)

def calculate_kl(p, q):
    # 0으로 나누기 및 log(0) 방지
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

# ==========================================
# 3. 사이드바 UI
# ==========================================
st.sidebar.title("🧪 지식 증류 패널")
st.sidebar.markdown("Teacher의 지식을 Student에게 전달!")

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ 1. Softmax 온도 조절")
temperature = st.sidebar.slider("온도 (Temperature, T)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 2. KL 발산 최적화 목표")
kl_type = st.sidebar.radio(
    "Student 모델의 학습 방향 선택",
    ["Forward KL (Mean-seeking)", "Reverse KL (Mode-seeking)"]
)

# ==========================================
# 4. 메인 화면: 탭 구성
# ==========================================
st.title("🎓 지식 증류(Knowledge Distillation) 시뮬레이터")

tab1, tab2 = st.tabs(["🌡️ 1. 소프트맥스 온도 조절 (Dark Knowledge)", "📉 2. Forward vs Reverse KL 발산"])

# ------------------------------------------
# 탭 1: 소프트맥스 온도 조절
# ------------------------------------------
with tab1:
    st.markdown("### 선생님 모델의 '숨겨진 지식' 추출하기")
    st.markdown("""
    일반적인 학습($T=1$)에서는 정답 확률만 압도적으로 높게 나옵니다. 하지만 온도($T$)를 높이면, 선생님 모델이 **"이 이미지가 개(Dog)일 확률이 가장 높지만, 고양이(Cat)와도 꽤 비슷하고, 자동차(Car)와는 전혀 안 닮았다"**고 생각하는 미세한 확률 분포(Dark Knowledge)가 드러납니다.
    """)
    
    # 가상의 Logits 데이터 (개, 고양이, 새, 자동차, 비행기)
    classes = ["개 (정답)", "고양이", "새", "자동차", "비행기"]
    logits = np.array([10.0, 5.0, 2.0, -2.0, -5.0])
    
    prob_t1 = softmax_with_temp(logits, 1.0)
    prob_t_custom = softmax_with_temp(logits, temperature)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        fig_t1 = go.Figure(data=[
            go.Bar(x=classes, y=prob_t1, marker_color='#1f77b4', text=[f"{p*100:.1f}%" for p in prob_t1], textposition='auto')
        ])
        fig_t1.update_layout(title="Hard Labels (T = 1.0)", template="plotly_dark", height=400, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig_t1, use_container_width=True)
        
    with col_t2:
        fig_t_custom = go.Figure(data=[
            go.Bar(x=classes, y=prob_t_custom, marker_color='#9467bd', text=[f"{p*100:.1f}%" for p in prob_t_custom], textposition='auto')
        ])
        fig_t_custom.update_layout(title=f"Soft Labels (T = {temperature:.1f}) - Dark Knowledge 발현!", template="plotly_dark", height=400, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig_t_custom, use_container_width=True)

# ------------------------------------------
# 탭 2: 정방향 vs 역방향 KL
# ------------------------------------------
with tab2:
    st.markdown("### 선생님(다형성)을 따라하는 학생(단일형)의 딜레마")
    st.markdown("""
    선생님 모델($P$)은 복잡한 데이터 분포(Bimodal, 두 개의 봉우리)를 완벽히 이해합니다. 하지만 학생 모델($Q$)은 용량이 작아 단순한 정규분포(Unimodal, 하나의 봉우리)만 표현할 수 있습니다. **이때 어떤 수학적 목표(KL Divergence)를 설정하느냐에 따라 학생의 학습 결과가 완전히 달라집니다.**
    """)
    
    # X 축 생성
    x = np.linspace(-10, 10, 500)
    
    # Teacher 분포 P (Bimodal: 두 가지 정답 가능성이 모두 높음)
    # 예: 언어 모델에서 "I am going to ___" (가능성: "sleep" 또는 "eat")
    p_dist = 0.5 * norm.pdf(x, -4, 1.5) + 0.5 * norm.pdf(x, 4, 1.5)
    p_dist = p_dist / np.sum(p_dist) # 정규화
    
# Student 분포 Q 세팅
    if kl_type == "Forward KL (Mean-seeking)":
        # Forward KL: P를 커버하려고 넓게 펴짐 (Zero-avoiding)
        q_dist = norm.pdf(x, 0, 4.5)
        q_dist = q_dist / np.sum(q_dist)
        kl_val = calculate_kl(p_dist, q_dist)
        formula_latex = r"D_{KL}(P_{teacher} \parallel Q_{student})"
        desc = "<b>Mean-seeking (Zero-avoiding)</b><br>학생 모델이 선생님의 모든 지식을 아우르려고 넓게 퍼집니다. 하지만 정작 가운데(확률이 가장 낮은 쓸모없는 곳)를 평균으로 잡아버려 애매하고 흐릿한 결과를 낼 수 있습니다. (전통적인 분류 문제의 KD 방식)"
        q_color = "#ff7f0e"
    else:
        # Reverse KL: 하나의 피크를 선택함 (Mode-seeking)
        q_dist = norm.pdf(x, 4, 1.5) # 오른쪽 봉우리 선택
        q_dist = q_dist / np.sum(q_dist)
        kl_val = calculate_kl(q_dist, p_dist)
        formula_latex = r"D_{KL}(Q_{student} \parallel P_{teacher})"
        desc = "<b>Mode-seeking (Zero-forcing)</b><br>학생 모델이 양쪽을 다 가지는 것을 포기하고, 선생님의 가장 확실한 한쪽 정답(봉우리)에 완벽하게 집중합니다. (최신 LLM, Diffusion 모델 등 생성형 AI에서 선명한 결과물을 내기 위해 선호됨)"
        q_color = "#2ca02c"

    # 그래프 그리기
    fig_kl = go.Figure()
    fig_kl.add_trace(go.Scatter(x=x, y=p_dist, name='Teacher (P)', fill='tozeroy', line=dict(color='#1f77b4', width=3), opacity=0.6))
    fig_kl.add_trace(go.Scatter(x=x, y=q_dist, name='Student (Q)', line=dict(color=q_color, width=4, dash='dash')))
    
    fig_kl.update_layout(
        template="plotly_dark", height=500,
        title=f"선택된 최적화: {kl_type}",
        xaxis_title="데이터 / 예측 범위", yaxis_title="확률 밀도 (Probability Density)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    col_k1, col_k2 = st.columns([1, 2])
    
    with col_k1:
        # 1. 제목
        st.markdown(f"<h3 style='color: {q_color}; margin-bottom: 0;'>💡 수식 및 분석</h3>", unsafe_allow_html=True)
        
        # 2. 수식 (Streamlit 전용 latex 함수 사용 -> 완벽하게 렌더링 됨!)
        st.latex(formula_latex)
        
        # 3. 설명 박스 (디자인 유지)
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 12px; border-left: 6px solid {q_color}; margin-top: 10px;">
            <p style="font-size: 16px; line-height: 1.6; color: #dddddd; margin-bottom: 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_k2:
        st.plotly_chart(fig_kl, use_container_width=True)

st.markdown("<br><div style='background:#2c2f33; padding:15px; border-radius:8px; text-align:center;'><b>made by song</b></div>", unsafe_allow_html=True)
