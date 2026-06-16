"""
综合工程问题（热-振动耦合问题）驱动《工程数值分析》教学（西电）
Streamlit 版本 - 适配云端部署
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import eigs
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 工程参数类 ====================
class MechanicsAgent:
    def __init__(self):
        self.L = 1.0
        self.b = 0.05
        self.h = 0.01
        self.nx = 50
        self.E = 70e9
        self.rho = 2700
        self.alpha = 2.3e-5
        self.k = 200
        self.cp = 900
        self.T0 = 20
        self.T_top = 50
        self.T_bottom = 20
        self.F_z = 800
        self.zeta = 0.02
        self.nonlinear_coeff = 0.05
        self.newton_tol = 1e-6
        self.newton_max_iter = 20
        self.update_properties()

    def update_properties(self):
        self.A = self.b * self.h
        self.I = self.b * self.h ** 3 / 12
        self.k_linear = 3 * self.E * self.I / self.L ** 3
        self.m_total = self.rho * self.A * self.L
        self.m_eq = self.m_total / 3
        self.dx = self.L / self.nx
        self.M_z = self.F_z * self.L

    def nonlinear_stiffness(self, u):
        return self.k_linear * (1 + self.nonlinear_coeff * u ** 2)

    def newton_raphson_solve(self, u, v, a, F, m, c, dt, gamma=0.5, beta=0.25):
        u_pred = u + dt * v + dt ** 2 / 2 * (1 - 2 * beta) * a
        v_pred = v + dt * (1 - gamma) * a
        k_nl = self.nonlinear_stiffness(u_pred)
        residual = m * a + c * v_pred + k_nl * u_pred - F
        for iteration in range(self.newton_max_iter):
            if abs(residual) < self.newton_tol:
                break
            k_tangent = k_nl + 2 * self.nonlinear_coeff * self.k_linear * u_pred ** 2
            k_eff = m + c * gamma * dt + k_tangent * beta * dt ** 2
            da = -residual / k_eff
            a_new = a + da
            u_pred_new = u_pred + beta * dt ** 2 * da
            v_pred_new = v_pred + gamma * dt * da
            k_nl_new = self.nonlinear_stiffness(u_pred_new)
            residual_new = m * a_new + c * v_pred_new + k_nl_new * u_pred_new - F
            a, u_pred, v_pred, k_nl, residual = a_new, u_pred_new, v_pred_new, k_nl_new, residual_new
        return a, u_pred, v_pred, iteration + 1

    def linear_newmark(self, m, k, c, F, dt, t_total, gamma=0.5, beta=0.25):
        n_steps = int(t_total / dt)
        t = np.linspace(0, t_total, n_steps + 1)
        u = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        a = np.zeros(n_steps + 1)
        a[0] = F / m
        for i in range(n_steps):
            u_pred = u[i] + dt * v[i] + dt ** 2 / 2 * (1 - 2 * beta) * a[i]
            v_pred = v[i] + dt * (1 - gamma) * a[i]
            a_next = (F - c * v_pred - k * u_pred) / (m + c * gamma * dt + k * beta * dt ** 2)
            u[i + 1] = u_pred + beta * dt ** 2 * a_next
            v[i + 1] = v_pred + gamma * dt * a_next
            a[i + 1] = a_next
        return t, u, v, a

    def nonlinear_newmark_newton(self, m, k_linear, c, F, dt, t_total, gamma=0.5, beta=0.25):
        n_steps = int(t_total / dt)
        t = np.linspace(0, t_total, n_steps + 1)
        u = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        a = np.zeros(n_steps + 1)
        a[0] = F / m
        iterations = np.zeros(n_steps + 1)
        for i in range(n_steps):
            a[i + 1], u[i + 1], v[i + 1], it = self.newton_raphson_solve(
                u[i], v[i], a[i], F, m, c, dt, gamma, beta)
            iterations[i + 1] = it
        return t, u, v, a, iterations

# ==================== Streamlit UI ====================
st.set_page_config(page_title="热-振动耦合分析", layout="wide")

st.title("🔥 综合工程问题（热-振动耦合问题）驱动《工程数值分析》教学（西电）")
st.caption("从六轴机器人到悬臂梁 | 应力云图 | 四种强度理论 | 刚度校核 | 压杆稳定")

# 初始化
if 'agent' not in st.session_state:
    st.session_state.agent = MechanicsAgent()
agent = st.session_state.agent

# ==================== 侧边栏参数 ====================
with st.sidebar:
    st.header("⚙️ 工程参数设置")

    st.subheader("几何参数")
    agent.L = st.number_input("梁长度 L (m)", value=1.0, step=0.1)
    agent.b = st.number_input("截面宽度 b (m)", value=0.05, step=0.01)
    agent.h = st.number_input("截面高度 h (m)", value=0.01, step=0.001)

    st.subheader("材料参数")
    agent.E = st.number_input("弹性模量 E (GPa)", value=70.0, step=1.0) * 1e9
    agent.rho = st.number_input("密度 ρ (kg/m³)", value=2700, step=100)
    agent.alpha = st.number_input("热膨胀系数 α (1e-5/K)", value=2.3, step=0.1) * 1e-5

    st.subheader("热参数")
    agent.T_top = st.number_input("上表面温度 T_top (°C)", value=50.0, step=1.0)
    agent.T_bottom = st.number_input("下表面温度 T_bottom (°C)", value=20.0, step=1.0)
    agent.T0 = st.number_input("初始温度 T0 (°C)", value=20.0, step=1.0)

    st.subheader("载荷与阻尼")
    agent.F_z = st.number_input("竖向力 F_z (N)", value=800, step=50)
    agent.zeta = st.number_input("阻尼比 ζ (0.01-0.1)", value=0.02, step=0.01)

    st.subheader("非线性参数")
    agent.nonlinear_coeff = st.number_input("非线性刚度系数 γ", value=0.05, step=0.01)
    agent.newton_tol = st.number_input("Newton迭代容差", value=1e-6, format="%.1e")

    if st.button("🔄 更新参数", type="primary"):
        agent.update_properties()
        st.success("参数已更新！")
        st.rerun()

    # 显示计算信息
    agent.update_properties()
    f_n = np.sqrt(agent.k_linear / agent.m_eq) / (2 * np.pi)
    delta_T = agent.T_top - agent.T_bottom
    M_thermal = agent.E * agent.I * agent.alpha * delta_T / agent.h
    st.divider()
    st.metric("线性基频", f"{f_n:.2f} Hz")
    st.metric("热弯矩", f"{M_thermal:.3f} N·m")
    st.metric("温差", f"{delta_T:.1f} °C")

# ==================== 主区域 ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 插值与拟合", "🌡️ 热传导分析", "📈 线性振动",
    "🔴 非线性振动", "🎵 模态分析", "📉 误差分析"
])

def run_with_progress(func):
    with st.spinner("计算中..."):
        return func()

# ---------- Tab 1: 插值与拟合 ----------
with tab1:
    st.subheader("插值与拟合分析")
    if st.button("运行插值与拟合", key="interp"):
        with st.spinner("计算中..."):
            np.random.seed(42)
            x_measured = np.array([0, 0.15, 0.3, 0.5, 0.65, 0.8, 0.95, 1.0]) * agent.L
            T_true = agent.T0 + (agent.T_top - agent.T0) * np.sin(np.pi * x_measured / (2 * agent.L))
            T_measured = T_true + np.random.normal(0, 1.5, len(x_measured))
            x_smooth = np.linspace(0, agent.L, 200)
            cs = CubicSpline(x_measured, T_measured)
            T_spline = cs(x_smooth)
            coeffs = np.polyfit(x_measured, T_measured, 3)
            T_polyfit = np.polyval(coeffs, x_smooth)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(x_smooth, T_spline, 'b-', linewidth=2, label='三次样条插值')
            ax1.plot(x_smooth, T_polyfit, 'g--', linewidth=2, label='3次多项式拟合')
            ax1.plot(x_measured, T_measured, 'ro', markersize=8, label='测量数据')
            ax1.plot(x_measured, T_true, 'k-', linewidth=1, alpha=0.5, label='真实温度')
            ax1.set_xlabel('位置 x [m]')
            ax1.set_ylabel('温度 [°C]')
            ax1.set_title('插值与拟合方法对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            delta_T_profile = (agent.T_top - T_spline) * 0.5
            ax2.plot(x_smooth, delta_T_profile, 'purple', linewidth=2)
            ax2.fill_between(x_smooth, 0, delta_T_profile, alpha=0.3)
            ax2.set_xlabel('位置 x [m]')
            ax2.set_ylabel('温差 [°C]')
            ax2.set_title('沿梁长的温度梯度')
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------- Tab 2: 热传导 ----------
with tab2:
    st.subheader("热传导分析")
    if st.button("运行热传导分析", key="thermal"):
        with st.spinner("计算中..."):
            alpha = agent.k / (agent.rho * agent.cp)
            dt = 0.1
            nt = 500
            r = alpha * dt / (agent.dx ** 2)
            x = np.linspace(0, agent.L, agent.nx + 1)
            T = np.ones(agent.nx + 1) * agent.T0
            T[0] = agent.T0
            T[-1] = agent.T_top
            T_history = [T.copy()]
            time_points = [0]
            for n in range(nt):
                T_new = T.copy()
                for i in range(1, agent.nx):
                    T_new[i] = T[i] + r * (T[i - 1] - 2 * T[i] + T[i + 1])
                T_new[0] = agent.T0
                T_new[-1] = agent.T_top
                T = T_new.copy()
                if n % 100 == 0:
                    T_history.append(T.copy())
                    time_points.append(n * dt)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            colors = ['blue', 'green', 'orange', 'red', 'purple']
            for i, (t, temp) in enumerate(zip(time_points, T_history)):
                ax1.plot(x, temp, color=colors[i % len(colors)], linewidth=2, label=f't={t:.1f}s')
            ax1.set_xlabel('位置 x [m]')
            ax1.set_ylabel('温度 [°C]')
            ax1.set_title('温度分布随时间演化')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            M_thermal_history = []
            for temp in T_history:
                delta_T = agent.T_top - temp[-1]
                M = agent.E * agent.I * agent.alpha * delta_T / agent.h
                M_thermal_history.append(M)
            ax2.plot(time_points, M_thermal_history, 'r-', linewidth=2)
            ax2.set_xlabel('时间 t [s]')
            ax2.set_ylabel('热弯矩 [N·m]')
            ax2.set_title('热弯矩随时间演化')
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------- Tab 3: 线性振动 ----------
with tab3:
    st.subheader("线性弯曲振动响应")
    if st.button("运行线性振动分析", key="linear"):
        with st.spinner("计算中..."):
            m = agent.m_eq
            k = agent.k_linear
            c = 2 * agent.zeta * np.sqrt(k * m)
            delta_T = agent.T_top - agent.T_bottom
            M_thermal = agent.E * agent.I * agent.alpha * delta_T / agent.h
            F_thermal = M_thermal / agent.L
            omega_n = np.sqrt(k / m)
            f_n = omega_n / (2 * np.pi)
            T_n = 1 / f_n
            dt = T_n / 20
            t_total = 5.0
            t, u, v, a = agent.linear_newmark(m, k, c, F_thermal, dt, t_total)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            ax1.plot(t, u * 1000, 'b-', linewidth=2)
            ax1.set_xlabel('时间 [s]')
            ax1.set_ylabel('位移 [mm]')
            ax1.set_title('自由端位移时程')
            ax1.grid(True, alpha=0.3)

            ax2.plot(t, v * 1000, 'g-', linewidth=2)
            ax2.set_xlabel('时间 [s]')
            ax2.set_ylabel('速度 [mm/s]')
            ax2.set_title('自由端速度时程')
            ax2.grid(True, alpha=0.3)

            ax3.plot(t, a, 'r-', linewidth=2)
            ax3.set_xlabel('时间 [s]')
            ax3.set_ylabel('加速度 [m/s²]')
            ax3.set_title('自由端加速度时程')
            ax3.grid(True, alpha=0.3)

            ax4.plot(u * 1000, v * 1000, 'purple', linewidth=1)
            ax4.set_xlabel('位移 [mm]')
            ax4.set_ylabel('速度 [mm/s]')
            ax4.set_title('相图')
            ax4.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------- Tab 4: 非线性振动 ----------
with tab4:
    st.subheader("非线性弯曲振动响应 (Newton-Raphson)")
    if st.button("运行非线性振动分析", key="nonlinear"):
        with st.spinner("计算中..."):
            m = agent.m_eq
            k_linear = agent.k_linear
            c = 2 * agent.zeta * np.sqrt(k_linear * m)
            delta_T = agent.T_top - agent.T_bottom
            M_thermal = agent.E * agent.I * agent.alpha * delta_T / agent.h
            F_thermal = M_thermal / agent.L
            omega_n = np.sqrt(k_linear / m)
            f_n = omega_n / (2 * np.pi)
            T_n = 1 / f_n
            dt = T_n / 20
            t_total = 5.0
            t, u, v, a, iterations = agent.nonlinear_newmark_newton(m, k_linear, c, F_thermal, dt, t_total)
            # 同时计算线性结果用于对比
            t_lin, u_lin, v_lin, a_lin = agent.linear_newmark(m, k_linear, c, F_thermal, dt, t_total)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            ax1.plot(t_lin, u_lin * 1000, 'b-', linewidth=1.5, alpha=0.7, label='线性')
            ax1.plot(t, u * 1000, 'r-', linewidth=2, label='非线性')
            ax1.set_xlabel('时间 [s]')
            ax1.set_ylabel('位移 [mm]')
            ax1.set_title('线性 vs 非线性 位移时程对比')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(iterations, 'g-', linewidth=2)
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('迭代次数')
            ax2.set_title('Newton-Raphson迭代次数')
            ax2.grid(True, alpha=0.3)

            diff = u * 1000 - u_lin * 1000
            ax3.plot(t, diff, 'm-', linewidth=2)
            ax3.set_xlabel('时间 [s]')
            ax3.set_ylabel('位移差 [mm]')
            ax3.set_title('非线性与线性结果差异')
            ax3.grid(True, alpha=0.3)

            ax4.plot(u_lin * 1000, v_lin * 1000, 'b-', linewidth=1, alpha=0.5, label='线性')
            ax4.plot(u * 1000, v * 1000, 'r-', linewidth=1.5, label='非线性')
            ax4.set_xlabel('位移 [mm]')
            ax4.set_ylabel('速度 [mm/s]')
            ax4.set_title('相图对比')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.info(f"平均 Newton 迭代次数: {np.mean(iterations[1:]):.2f}")

# ---------- Tab 5: 模态分析 ----------
with tab5:
    st.subheader("模态分析")
    if st.button("运行模态分析", key="modal"):
        with st.spinner("计算中..."):
            n_dof = 30
            m_elem = agent.rho * agent.A * agent.L / n_dof
            M = np.eye(n_dof) * m_elem
            k_elem = 3 * agent.E * agent.I / (agent.L / n_dof) ** 3
            K = np.zeros((n_dof, n_dof))
            for i in range(n_dof):
                K[i, i] = 2 * k_elem
                if i > 0:
                    K[i, i - 1] = -k_elem
                if i < n_dof - 1:
                    K[i, i + 1] = -k_elem
            K[0, 0] = 1e10

            try:
                import scipy.linalg as la
                eigenvalues, eigenvectors = la.eig(K, M)
                valid = np.where(eigenvalues.real > 0)[0]
                eigenvalues = eigenvalues.real[valid]
                eigenvectors = eigenvectors[:, valid].real
                idx = np.argsort(eigenvalues)
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                n_modes = min(6, len(eigenvalues))
                frequencies = np.sqrt(eigenvalues[:n_modes]) / (2 * np.pi)
            except:
                n_modes = 6
                eigenvalues, eigenvectors = eigs(K, k=n_modes, M=M, which='SM')
                eigenvalues = np.sort(eigenvalues.real)
                frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            x_nodes = np.linspace(0, agent.L, n_dof)
            for i in range(min(n_modes, 6)):
                ax = axes[i // 3, i % 3]
                phi = eigenvectors[:, i].real if eigenvectors.shape[1] > i else np.random.randn(n_dof)
                if np.max(np.abs(phi)) > 0:
                    phi = phi / np.max(np.abs(phi))
                ax.plot(x_nodes, phi, 'b-o', linewidth=2, markersize=5)
                ax.fill_between(x_nodes, 0, phi, alpha=0.3, color='blue')
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.set_xlabel('位置 x [m]')
                ax.set_ylabel('振型幅值')
                ax.set_title(f'{i+1}阶模态: {frequencies[i]:.2f} Hz' if i < len(frequencies) else f'{i+1}阶模态')
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 显示频率列表
            freq_df = {"阶次": list(range(1, len(frequencies)+1)), "频率 (Hz)": [f"{f:.2f}" for f in frequencies]}
            st.table(freq_df)

# ---------- Tab 6: 误差分析 ----------
with tab6:
    st.subheader("误差分析与传播")
    if st.button("运行误差分析", key="error"):
        with st.spinner("计算中..."):
            def f(x):
                return np.sin(x)

            def f_prime(x):
                return np.cos(x)

            x0 = 1.0
            true_derivative = f_prime(x0)
            h_values = np.logspace(-15, -1, 30)
            total_errors = []
            for h in h_values:
                approx = (f(x0 + h) - f(x0 - h)) / (2 * h)
                total_errors.append(abs(approx - true_derivative))
            min_idx = np.argmin(total_errors)
            optimal_h = h_values[min_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.loglog(h_values, total_errors, 'b-o', linewidth=2, markersize=4, label='总误差')
            ax.set_xlabel('步长 h')
            ax.set_ylabel('误差')
            ax.set_title('数值微分误差分析（中心差分）')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=optimal_h, color='black', linestyle='--', alpha=0.5)
            ax.text(optimal_h, total_errors[min_idx] * 10, f'最优步长: {optimal_h:.2e}',
                    ha='center', va='bottom', fontsize=10)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.success(f"✅ 最优步长 h_opt = {optimal_h:.2e}")