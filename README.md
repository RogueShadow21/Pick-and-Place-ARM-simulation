# AI-Based Adaptive Control of Pneumatic Pick & Place Robotic Arm

Interactive Streamlit simulation comparing a **Fixed-Gain PID Controller** against an **AI-Based Online RBF Neural Network Adaptive Controller** on a 2-Link Articulated Pneumatic Robotic Arm.

## Features
- 2-Link Euler-Lagrange robotic arm driven by two ISO-6358 compressible-flow pneumatic cylinders
- Stribeck + Coulomb + Viscous joint friction
- Sudden payload changes (Cycle 1, Cycle 2 heavy load + air pressure drop, Cycle 3)
- Online RBF Neural Network feedforward compensator + self-tuning gain Kp(t)
- Interactive Streamlit UI with Play/Pause animation, KPI cards, chamber pressures, and CSV export

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```
