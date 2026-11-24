import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

st.set_page_config(
    page_title="AI Adaptive Pneumatic Robotic Arm",
    page_icon="🤖",
    layout="wide",
)

class ArticulatedPneumaticArm:
    def __init__(self, bore=0.040, rod=0.016, P_atm=101325.0):
        self.x0, self.z0 = 0.0, 0.12
        self.L1, self.L2 = 0.45, 0.40
        self.m1, self.m2 = 2.4, 1.6
        self.lc1, self.lc2 = 0.225, 0.20
        self.I1 = (1.0 / 12.0) * self.m1 * self.L1**2
        self.I2 = (1.0 / 12.0) * self.m2 * self.L2**2
        self.g = 9.81

        self.a1, self.b1, self.alpha1 = 0.18, 0.22, np.radians(-45.0)
        self.a2, self.b2 = 0.24, 0.14

        self.c1_min = self._cyl_len(np.radians(20.0), self.a1, self.b1, -self.alpha1)
        self.c1_max = self._cyl_len(np.radians(105.0), self.a1, self.b1, -self.alpha1)
        self.S1 = max(self.c1_max - self.c1_min, 0.20)

        self.c2_min = self._cyl_len(np.radians(-150.0), self.a2, self.b2, np.pi)
        self.c2_max = self._cyl_len(np.radians(-40.0), self.a2, self.b2, np.pi)
        self.S2 = max(self.c2_max - self.c2_min, 0.20)

        self.A_A = np.pi * (bore / 2.0)**2
        self.A_B = self.A_A - np.pi * (rod / 2.0)**2
        self.V0 = 3.0e-5
        self.gamma = 1.4
        self.R = 287.0
        self.P_atm = P_atm
        self.Av_max = 1.6e-5
        self.Cd = 0.82
        self.b_cr = 0.528

        self.B_j = np.array([1.2, 0.8])
        self.Fc_j = np.array([2.2, 1.5])
        self.Fs_j = np.array([3.5, 2.4])
        self.reset()

    @staticmethod
    def _cyl_len(th, a, b, offset):
        psi = np.clip(th + offset, 0.15, np.pi - 0.15)
        return np.sqrt(a**2 + b**2 - 2.0 * a * b * np.cos(psi))

    @staticmethod
    def _moment_arm(th, a, b, offset):
        psi = np.clip(th + offset, 0.15, np.pi - 0.15)
        c = np.sqrt(a**2 + b**2 - 2.0 * a * b * np.cos(psi))
        J = (a * b * np.sin(psi)) / c
        return c, J

    def ik(self, x, z):
        dx, dz = x - self.x0, z - self.z0
        r2 = dx**2 + dz**2
        c2 = np.clip((r2 - self.L1**2 - self.L2**2) / (2.0 * self.L1 * self.L2), -0.995, 0.995)
        th2 = -np.arccos(c2)
        th1 = np.arctan2(dz, dx) - np.arctan2(self.L2 * np.sin(th2), self.L1 + self.L2 * c2)
        return th1, th2

    def fk(self, th1, th2):
        x1 = self.x0 + self.L1 * np.cos(th1)
        z1 = self.z0 + self.L1 * np.sin(th1)
        x2 = x1 + self.L2 * np.cos(th1 + th2)
        z2 = z1 + self.L2 * np.sin(th1 + th2)
        return (self.x0, self.z0), (x1, z1), (x2, z2)

    def cylinder_endpoints(self, th1, th2):
        A1 = (self.x0 + self.a1 * np.cos(self.alpha1), self.z0 + self.a1 * np.sin(self.alpha1))
        B1 = (self.x0 + self.b1 * np.cos(th1), self.z0 + self.b1 * np.sin(th1))
        _, p_elbow, _ = self.fk(th1, th2)
        A2 = (p_elbow[0] - self.a2 * np.cos(th1), p_elbow[1] - self.a2 * np.sin(th1))
        B2 = (p_elbow[0] + self.b2 * np.cos(th1 + th2), p_elbow[1] + self.b2 * np.sin(th1 + th2))
        return (A1, B1), (A2, B2)

    def reset(self, x_init=0.30, z_init=0.34):
        th1, th2 = self.ik(x_init, z_init)
        self.th = np.array([th1, th2])
        self.dth = np.array([0.0, 0.0])
        self.PA = np.array([3.2e5, 3.0e5])
        self.PB = np.array([2.8e5, 3.0e5])

    def orifice_flow(self, P_up, P_down, area, T):
        if area <= 1e-12:
            return 0.0
        sign = 1.0
        if P_up < P_down:
            P_up, P_down = P_down, P_up
            sign = -1.0
        ratio = max(1e-4, min(1.0, P_down / max(P_up, 1e3)))
        if ratio <= self.b_cr:
            factor = np.sqrt((self.gamma / (self.R * T)) *
                             (2.0 / (self.gamma + 1.0)) ** ((self.gamma + 1.0) / (self.gamma - 1.0)))
        else:
            term = max(0.0, ratio**(2.0 / self.gamma) - ratio**((self.gamma + 1.0) / self.gamma))
            factor = np.sqrt((2.0 * self.gamma / (self.R * T * (self.gamma - 1.0))) * term)
        return sign * self.Cd * area * P_up * factor

    def manipulator_matrices(self, th1, th2, dth1, dth2, m_load):
        m2_eff = self.m2 + m_load
        lc2_eff = (self.m2 * self.lc2 + m_load * self.L2) / m2_eff
        I2_eff = self.I2 + self.m2 * (self.lc2 - lc2_eff)**2 + m_load * (self.L2 - lc2_eff)**2

        c2, s2 = np.cos(th2), np.sin(th2)
        M11 = self.I1 + self.m1 * self.lc1**2 + I2_eff + m2_eff * (self.L1**2 + lc2_eff**2 + 2.0 * self.L1 * lc2_eff * c2)
        M12 = I2_eff + m2_eff * (lc2_eff**2 + self.L1 * lc2_eff * c2)
        M22 = I2_eff + m2_eff * lc2_eff**2
        M = np.array([[M11, M12], [M12, M22]])

        h = -m2_eff * self.L1 * lc2_eff * s2
        C = np.array([h * dth2 * (2.0 * dth1 + dth2), -h * dth1**2])
        G1 = (self.m1 * self.lc1 + m2_eff * self.L1) * self.g * np.cos(th1) + m2_eff * lc2_eff * self.g * np.cos(th1 + th2)
        G2 = m2_eff * lc2_eff * self.g * np.cos(th1 + th2)
        return M, C, np.array([G1, G2])

    def derivatives(self, state, u, m_load, P_supply, friction_scale, T):
        th1, th2, dth1, dth2, PA1, PA2, PB1, PB2 = state
        u1, u2 = np.clip(u[0], -1.0, 1.0), np.clip(u[1], -1.0, 1.0)

        c1, J1 = self._moment_arm(th1, self.a1, self.b1, -self.alpha1)
        c2, J2 = self._moment_arm(th2, self.a2, self.b2, np.pi)
        s1 = np.clip(c1 - self.c1_min, 0.0, self.S1)
        s2 = np.clip(c2 - self.c2_min, 0.0, self.S2)
        ds1, ds2 = J1 * dth1, J2 * dth2

        dP_A, dP_B = np.zeros(2), np.zeros(2)
        for i, (si, dsi, Si, PAi, PBi, ui) in enumerate([
            (s1, ds1, self.S1, PA1, PB1, u1),
            (s2, ds2, self.S2, PA2, PB2, u2)
        ]):
            VA = self.V0 + self.A_A * si
            VB = self.V0 + self.A_B * (Si - si)
            if ui >= 0.0:
                area = ui * self.Av_max
                mA = self.orifice_flow(P_supply, PAi, area, T)
                mB = -self.orifice_flow(PBi, self.P_atm, area, T)
            else:
                area = (-ui) * self.Av_max
                mA = -self.orifice_flow(PAi, self.P_atm, area, T)
                mB = self.orifice_flow(P_supply, PBi, area, T)
            dP_A[i] = (self.gamma * self.R * T / VA) * mA - (self.gamma * PAi * self.A_A / VA) * dsi
            dP_B[i] = (self.gamma * self.R * T / VB) * mB + (self.gamma * PBi * self.A_B / VB) * dsi

        F_cyl1 = PA1 * self.A_A - PB1 * self.A_B - self.P_atm * (self.A_A - self.A_B)
        F_cyl2 = PA2 * self.A_A - PB2 * self.A_B - self.P_atm * (self.A_A - self.A_B)
        tau_pneu = np.array([F_cyl1 * J1, F_cyl2 * J2])

        dth = np.array([dth1, dth2])
        tau_fric = friction_scale * (
            self.B_j * dth +
            (self.Fc_j + (self.Fs_j - self.Fc_j) * np.exp(-(dth / 0.08)**2)) * np.tanh(dth / 0.03)
        )

        M, C, G = self.manipulator_matrices(th1, th2, dth1, dth2, m_load)
        ddth = np.linalg.solve(M, tau_pneu - C - G - tau_fric)
        return np.array([dth1, dth2, ddth[0], ddth[1], dP_A[0], dP_A[1], dP_B[0], dP_B[1]])

    def step(self, u, m_load, P_supply, dt, friction_scale=1.0, T=293.15):
        s = np.array([self.th[0], self.th[1], self.dth[0], self.dth[1],
                      self.PA[0], self.PA[1], self.PB[0], self.PB[1]])
        k1 = self.derivatives(s, u, m_load, P_supply, friction_scale, T)
        k2 = self.derivatives(s + 0.5 * dt * k1, u, m_load, P_supply, friction_scale, T)
        k3 = self.derivatives(s + 0.5 * dt * k2, u, m_load, P_supply, friction_scale, T)
        k4 = self.derivatives(s + dt * k3, u, m_load, P_supply, friction_scale, T)
        sn = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.th = np.array([np.clip(sn[0], np.radians(10.0), np.radians(125.0)),
                            np.clip(sn[1], np.radians(-155.0), np.radians(-20.0))])
        self.dth = sn[2:4]
        self.PA = np.clip(sn[4:6], 0.5 * self.P_atm, 1.5 * P_supply)
        self.PB = np.clip(sn[6:8], 0.5 * self.P_atm, 1.5 * P_supply)
        return self.th, self.dth, self.PA, self.PB


class JointPID:
    def __init__(self, kp=2.2, ki=1.0, kd=0.22):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0

    def compute(self, ref, dref, pos, vel, dt):
        e = ref - pos
        de = dref - vel
        u_raw = self.kp * e + self.ki * self.integral + self.kd * de
        if abs(u_raw) < 1.0 or (u_raw * e < 0.0):
            self.integral = np.clip(self.integral + e * dt, -0.5, 0.5)
        u = np.clip(self.kp * e + self.ki * self.integral + self.kd * de, -1.0, 1.0)
        return u, 0.0, self.kp


class JointAIRBFController:
    def __init__(self, kp0=2.2, kd0=0.22, lam=10.0, lr_w=3.8, lr_gain=250.0):
        self.kp0, self.kd0 = kp0, kd0
        self.kp, self.kd = kp0, kd0
        self.lam = lam
        self.lr_w = lr_w
        self.lr_gain = lr_gain

        rng = np.random.default_rng(7)
        self.n_rbf = 24
        self.centers = rng.uniform(-1.0, 1.0, size=(self.n_rbf, 4))
        self.width = 0.75
        self.weights = np.zeros(self.n_rbf + 1)

    def features(self, e, de, ref, vel):
        z = np.array([
            np.clip(e / 0.35, -1.5, 1.5),
            np.clip(de / 1.5, -1.5, 1.5),
            np.clip(ref / 1.5, -1.5, 1.5),
            np.tanh(vel / 0.15),
        ])
        dists = np.sum((self.centers - z)**2, axis=1)
        return np.concatenate(([1.0], np.exp(-dists / (2.0 * self.width**2))))

    def compute(self, ref, dref, pos, vel, dt):
        e = ref - pos
        de = dref - vel
        s = de + self.lam * e

        phi = self.features(e, de, ref, vel)
        u_nn = float(np.dot(self.weights, phi))
        u = np.clip(self.kp * e + self.kd * de + u_nn, -1.0, 1.0)

        if abs(u) < 0.99 or (s * u < 0.0):
            dW = self.lr_w * s * phi - 0.12 * self.weights
            self.weights = np.clip(self.weights + dW * dt, -0.85, 0.85)

        dkp = self.lr_gain * (e * s) - 1.5 * (self.kp - self.kp0)
        dkd = 0.25 * self.lr_gain * (de * s) - 1.5 * (self.kd - self.kd0)
        self.kp = np.clip(self.kp + dkp * dt, 1.5, 6.5)
        self.kd = np.clip(self.kd + dkd * dt, 0.12, 0.8)
        return u, u_nn, self.kp


def min_jerk(p0, p1, tau):
    tau = np.clip(tau, 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return p0 + (p1 - p0) * s


def get_arm_mission(t, payloads, p_nom, p_drop):
    cycle_len = 4.5
    cyc = min(int(t // cycle_len), 2)
    tc = t - cyc * cycle_len
    P_supply = (p_drop * 1e5) if cyc == 1 else (p_nom * 1e5)

    x_pick, x_place = 0.30, 0.68
    z_down, z_up = 0.04, 0.34
    gripped = (0.75 <= tc <= 3.15)
    m_load = payloads[cyc] if gripped else 0.0

    if tc < 0.6:
        xr, zr = x_pick, min_jerk(z_up, z_down, tc / 0.6)
    elif tc < 0.9:
        xr, zr = x_pick, z_down
    elif tc < 1.5:
        xr, zr = x_pick, min_jerk(z_down, z_up, (tc - 0.9) / 0.6)
    elif tc < 2.4:
        xr, zr = min_jerk(x_pick, x_place, (tc - 1.5) / 0.9), z_up
    elif tc < 3.0:
        xr, zr = x_place, min_jerk(z_up, z_down, (tc - 2.4) / 0.6)
    elif tc < 3.3:
        xr, zr = x_place, z_down
    elif tc < 3.8:
        xr, zr = x_place, min_jerk(z_down, z_up, (tc - 3.3) / 0.5)
    else:
        xr, zr = min_jerk(x_place, x_pick, (tc - 3.8) / 0.7), z_up

    return xr, zr, m_load, P_supply, gripped, cyc


@st.cache_data(show_spinner="Running nonlinear pneumatic physics & AI adaptation...")
def simulate_both(m1, m2, m3, p_nom, p_drop, friction_scale, temp_c, lr_w, lr_gain):
    payloads = (m1, m2, m3)
    T_kelvin = temp_c + 273.15
    T_total, dt, stride = 13.5, 0.0008, 25
    steps = int(T_total / dt)
    n_save = steps // stride

    results = {}
    for mode in ["PID", "AI"]:
        arm = ArticulatedPneumaticArm()
        arm.reset(x_init=0.30, z_init=0.34)
        if mode == "PID":
            c1 = JointPID(kp=2.2, ki=1.0, kd=0.22)
            c2 = JointPID(kp=2.0, ki=0.9, kd=0.20)
        else:
            c1 = JointAIRBFController(kp0=2.2, kd0=0.22, lam=10.0, lr_w=lr_w, lr_gain=lr_gain)
            c2 = JointAIRBFController(kp0=2.0, kd0=0.20, lam=10.0, lr_w=lr_w * 0.85, lr_gain=lr_gain * 0.85)

        hist = {k: np.zeros(n_save) for k in [
            "t", "x", "z", "xr", "zr", "th1", "th2", "th1_r", "th2_r",
            "PA1", "PB1", "PA2", "PB2", "unn1", "kp1", "m_load", "P_s", "gripped", "cyc"
        ]}
        prev_th_r = np.array(arm.ik(0.30, 0.34))
        idx = 0
        for k in range(steps):
            t = k * dt
            xr, zr, m_load, P_s, gripped, cyc = get_arm_mission(t, payloads, p_nom, p_drop)
            th_r = np.array(arm.ik(xr, zr))
            dth_r = (th_r - prev_th_r) / dt if k > 0 else np.zeros(2)
            prev_th_r = th_r

            u1, unn1, kp1 = c1.compute(th_r[0], dth_r[0], arm.th[0], arm.dth[0], dt)
            u2, _, _ = c2.compute(th_r[1], dth_r[1], arm.th[1], arm.dth[1], dt)

            th, _, PA, PB = arm.step([u1, u2], m_load, P_s, dt, friction_scale, T_kelvin)
            _, _, (x_ee, z_ee) = arm.fk(th[0], th[1])

            if k % stride == 0 and idx < n_save:
                hist["t"][idx] = t
                hist["x"][idx], hist["z"][idx] = x_ee, z_ee
                hist["xr"][idx], hist["zr"][idx] = xr, zr
                hist["th1"][idx], hist["th2"][idx] = th[0], th[1]
                hist["th1_r"][idx], hist["th2_r"][idx] = th_r[0], th_r[1]
                hist["PA1"][idx], hist["PB1"][idx] = PA[0] / 1e5, PB[0] / 1e5
                hist["PA2"][idx], hist["PB2"][idx] = PA[1] / 1e5, PB[1] / 1e5
                hist["unn1"][idx], hist["kp1"][idx] = unn1, kp1
                hist["m_load"][idx], hist["P_s"][idx] = m_load, P_s / 1e5
                hist["gripped"][idx], hist["cyc"][idx] = float(gripped), cyc
                idx += 1
        results[mode] = hist

    return results["PID"], results["AI"]


st.title("🤖 AI-Based Adaptive Control of a Pneumatic Pick & Place Robotic Arm")
st.caption("2-Link Articulated Robotic Arm driven by ISO-6358 Nonlinear Pneumatic Cylinders | Online RBF Neural Network vs. Fixed PID")

with st.sidebar:
    st.header("⚙️ Plant & Disturbance Controls")
    m1 = st.slider("Cycle 1 Payload (kg)", 0.1, 3.0, 0.4, 0.1)
    m2 = st.slider("Cycle 2 Heavy Payload (kg)", 1.0, 6.0, 4.2, 0.1)
    m3 = st.slider("Cycle 3 Payload (kg)", 0.5, 5.0, 2.5, 0.1)

    p_nom = st.slider("Nominal Air Supply (bar)", 5.0, 8.0, 6.0, 0.2)
    p_drop = st.slider("Cycle 2 Pressure Drop (bar)", 3.5, 6.5, 4.6, 0.1)
    friction_scale = st.slider("Cylinder Friction Multiplier", 0.5, 2.0, 1.0, 0.1)
    temp_c = st.slider("Air Temperature (°C)", 5, 50, 20, 1)

    st.divider()
    st.header("🧠 AI Controller Hyperparameters")
    lr_w = st.slider("RBF Weight Learning Rate (Γ_w)", 0.5, 8.0, 3.8, 0.1)
    lr_gain = st.slider("Adaptive Gain Rate (Γ_kp)", 0.0, 400.0, 250.0, 10.0)

    st.divider()
    show_pid_ghost = st.checkbox("Show Fixed PID Ghost Arm", value=True)
    play_speed = st.select_slider("Animation Step Stride", options=[2, 4, 6, 10], value=4)

h_pid, h_ai = simulate_both(m1, m2, m3, p_nom, p_drop, friction_scale, temp_c, lr_w, lr_gain)
arm_helper = ArticulatedPneumaticArm()

err_pid = np.hypot(h_pid["xr"] - h_pid["x"], h_pid["zr"] - h_pid["z"]) * 1000.0
err_ai = np.hypot(h_ai["xr"] - h_ai["x"], h_ai["zr"] - h_ai["z"]) * 1000.0
rmse_pid = float(np.sqrt(np.mean(err_pid**2)))
rmse_ai = float(np.sqrt(np.mean(err_ai**2)))
imp_pct = (1.0 - rmse_ai / max(rmse_pid, 1e-6)) * 100.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("AI Adaptive RMSE", f"{rmse_ai:.2f} mm", delta=f"-{rmse_pid - rmse_ai:.2f} mm vs PID")
k2.metric("Fixed PID RMSE", f"{rmse_pid:.2f} mm")
k3.metric("Accuracy Improvement", f"{imp_pct:.1f} %")
k4.metric("Peak Adaptive Kp(t)", f"{h_ai['kp1'].max():.2f}", delta=f"+{h_ai['kp1'].max() - 2.20:.2f} adapted")

tab1, tab2, tab3, tab4 = st.tabs([
    "🦾 Live Robotic Arm Simulation",
    "📈 Trajectory & Error Telemetry",
    "🧠 Pneumatic Pressures & AI Learning",
    "📄 Export Simulation Data"
])

def draw_arm_frame(frame_idx):
    th1, th2 = h_ai["th1"][frame_idx], h_ai["th2"][frame_idx]
    th1_p, th2_p = h_pid["th1"][frame_idx], h_pid["th2"][frame_idx]
    xr, zr = h_ai["xr"][frame_idx], h_ai["zr"][frame_idx]
    gripped = h_ai["gripped"][frame_idx] > 0.5
    cyc = int(h_ai["cyc"][frame_idx])
    loads = [m1, m2, m3]

    p0, p1, p2 = arm_helper.fk(th1, th2)
    _, p1_p, p2_p = arm_helper.fk(th1_p, th2_p)
    (A1, B1), (A2, B2) = arm_helper.cylinder_endpoints(th1, th2)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.set_xlim(-0.18, 0.82)
    ax.set_ylim(-0.08, 0.72)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    ax.plot([-0.15, 0.80], [0.0, 0.0], "k-", lw=3)
    ax.add_patch(patches.Rectangle((-0.07, 0.0), 0.14, 0.12, color="#37474f"))
    ax.add_patch(patches.Rectangle((0.24, -0.025), 0.12, 0.025, color="#555555"))
    ax.add_patch(patches.Rectangle((0.62, -0.025), 0.12, 0.025, color="#2e7d32"))
    ax.text(0.30, -0.06, "PICK STATION", ha="center", fontsize=8.5, fontweight="bold")
    ax.text(0.68, -0.06, "PLACE STATION", ha="center", fontsize=8.5, fontweight="bold", color="#2e7d32")

    if show_pid_ghost:
        ax.plot([p0[0], p1_p[0], p2_p[0]], [p0[1], p1_p[1], p2_p[1]], "r--", lw=2.2, alpha=0.55, label="Fixed PID (Ghost)")
        ax.plot(p2_p[0], p2_p[1], "ro", ms=6, alpha=0.55)

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#1565c0", lw=7.5, solid_capstyle="round", label="AI Adaptive Arm")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1e88e5", lw=6.0, solid_capstyle="round")

    for A, B in [(A1, B1), (A2, B2)]:
        mx = A[0] + 0.55 * (B[0] - A[0])
        mz = A[1] + 0.55 * (B[1] - A[1])
        ax.plot([A[0], mx], [A[1], mz], color="#ff8f00", lw=5.5, solid_capstyle="butt")
        ax.plot([mx, B[0]], [mz, B[1]], color="#b0bec5", lw=2.5)
        ax.plot([A[0], B[0]], [A[1], B[1]], "ko", ms=3.5)

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "ko", ms=6.5)
    ax.plot(p2[0], p2[1], "bs", ms=7.5)
    ax.plot(xr, zr, "kx", ms=9.5, mew=2.2, label="Target")

    tc = h_ai["t"][frame_idx] - cyc * 4.5
    if gripped:
        bx = (p2[0] - 0.025, p2[1] - 0.05)
    elif tc < 0.75:
        bx = (0.275, 0.0)
    else:
        bx = (0.655, 0.0)
    ax.add_patch(patches.Rectangle(bx, 0.05, 0.05, facecolor="orange", edgecolor="black", lw=1.2))

    e_ai_mm = np.hypot(xr - p2[0], zr - p2[1]) * 1000.0
    e_pid_mm = np.hypot(xr - p2_p[0], zr - p2_p[1]) * 1000.0
    ax.set_title(
        f"t={h_ai['t'][frame_idx]:.2f}s | Cycle {cyc+1}/3 | Payload={loads[cyc]:.1f}kg | "
        f"Ps={h_ai['P_s'][frame_idx]:.1f}bar | AI Err={e_ai_mm:.1f}mm | PID Err={e_pid_mm:.1f}mm",
        fontsize=10, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig

with tab1:
    c_play, c_scrub = st.columns([1, 4])
    if "playing" not in st.session_state:
        st.session_state.playing = False

    with c_play:
        if st.button("▶️ Play / ⏸️ Pause", use_container_width=True):
            st.session_state.playing = not st.session_state.playing

    with c_scrub:
        frame_idx = st.slider("Timeline Scrubber (frame)", 0, len(h_ai["t"]) - 1, 285, 1)

    anim_slot = st.empty()
    if st.session_state.playing:
        for f in range(frame_idx, len(h_ai["t"]), play_speed):
            if not st.session_state.playing:
                break
            fig = draw_arm_frame(f)
            anim_slot.pyplot(fig, use_container_width=True)
            plt.close(fig)
            time.sleep(0.03)
        st.session_state.playing = False
    else:
        fig = draw_arm_frame(frame_idx)
        anim_slot.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab2:
    fig_t, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = h_ai["t"]

    axs[0].plot(t, h_ai["xr"] * 1000, "k--", lw=1.4, label="Reference X")
    axs[0].plot(t, h_pid["x"] * 1000, "r-", alpha=0.75, label="Fixed PID")
    axs[0].plot(t, h_ai["x"] * 1000, "b-", lw=1.5, label="AI Adaptive")
    axs[0].set_ylabel("X (mm)")
    axs[0].set_title("End-Effector Horizontal Position (X)")
    axs[0].legend(loc="upper right", ncol=3)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, h_ai["zr"] * 1000, "k--", lw=1.4, label="Reference Z")
    axs[1].plot(t, h_pid["z"] * 1000, "r-", alpha=0.75, label="Fixed PID")
    axs[1].plot(t, h_ai["z"] * 1000, "b-", lw=1.5, label="AI Adaptive")
    axs[1].set_ylabel("Z (mm)")
    axs[1].set_title("End-Effector Vertical Position (Z)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t, err_pid, "r-", alpha=0.75, label="Fixed PID Error")
    axs[2].plot(t, err_ai, "b-", lw=1.5, label="AI Adaptive Error")
    axs[2].set_ylabel("Error (mm)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Euclidean End-Effector Tracking Error (mm)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)

with tab3:
    fig_p, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = h_ai["t"]

    axs[0].plot(t, h_ai["PA1"], label="Shoulder Chamber A (bar)")
    axs[0].plot(t, h_ai["PB1"], label="Shoulder Chamber B (bar)")
    axs[0].plot(t, h_ai["P_s"], "k--", lw=1.2, label="Supply Pressure (bar)")
    axs[0].set_ylabel("Pressure (bar)")
    axs[0].set_title("Shoulder Pneumatic Cylinder Chamber Pressures")
    axs[0].legend(loc="upper right", ncol=3)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, h_ai["unn1"], "g-", lw=1.5, label="RBF Neural Feedforward u_NN (Shoulder)")
    axs[1].set_ylabel("Valve Command")
    axs[1].set_title("Online AI Neural Network Compensation Signal")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    ax_r = axs[2].twinx()
    axs[2].plot(t, h_ai["kp1"], "m-", lw=1.5, label="Adaptive Kp(t)")
    ax_r.fill_between(t, 0, h_ai["m_load"], color="orange", alpha=0.25, label="Gripped Payload (kg)")
    axs[2].set_ylabel("Adaptive Kp", color="m")
    ax_r.set_ylabel("Payload (kg)", color="darkorange")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Self-Tuning Gain Kp(t) vs. Gripped Payload Disturbances")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_p, use_container_width=True)
    plt.close(fig_p)

with tab4:
    df = pd.DataFrame({
        "time_s": h_ai["t"],
        "x_ref_m": h_ai["xr"],
        "z_ref_m": h_ai["zr"],
        "x_ai_m": h_ai["x"],
        "z_ai_m": h_ai["z"],
        "x_pid_m": h_pid["x"],
        "z_pid_m": h_pid["z"],
        "err_ai_mm": err_ai,
        "err_pid_mm": err_pid,
        "u_nn_shoulder": h_ai["unn1"],
        "kp_adaptive": h_ai["kp1"],
        "payload_kg": h_ai["m_load"],
        "supply_bar": h_ai["P_s"],
    })
    st.dataframe(df.head(50), use_container_width=True)
    st.download_button(
        "⬇️ Download Full Telemetry CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pneumatic_arm_ai_telemetry.csv",
        mime="text/csv",
    )
