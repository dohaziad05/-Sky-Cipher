# 🚁 Sky Cipher: Drone Resilience & Mission Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Reinforcement Learning](https://img.shields.io/badge/RL-Reinforcement_Learning-red?style=for-the-badge)
![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

**Sky Cipher** is an advanced AI-driven simulation platform designed to predict and evaluate drone mission success in highly contested environments. By integrating **Reinforcement Learning (RL)** with high-fidelity physics simulations, Sky Cipher analyzes how drones can maintain operational integrity during **Electronic Warfare (EW)** and severe **Signal Jamming** scenarios.

## 🛡️ The Mission: Defeating the Jamming
In modern conflict zones, drones face constant threats from GPS denial and communication interference. **Sky Cipher** provides a predictive framework to:
* **Simulate Hostile Environments:** Model complex signal interference patterns.
* **Predict Mission Success:** Use AI to forecast whether a drone will reach its target or fail due to jamming.
* **Adaptive Behavior:** Train RL agents to develop "evasive" maneuvers or alternative navigation logic when primary signals are lost.

## 🧠 Technical Core
* **Physics Engine:** Powered by `PyBullet` for realistic 6-DOF drone dynamics.
* **AI Architecture:** Deep Reinforcement Learning agents trained to navigate through high-interference zones.
* **Data-Driven Insights:** Includes the `SkyCipher Dataset`, a comprehensive collection of flight frames categorized by mission outcomes (Evading, Crash, Success).
* **Predictive Dashboard:** A `Streamlit` interface for real-time visualization of mission success probabilities.

## 📂 Repository Structure
* 🛠️ `environment.py`: The core simulation arena featuring jamming variables and physics constraints.
* 🚀 `train.py` & `test.py`: Scripts for training the neural networks and evaluating mission resilience.
* 📊 `app.py`: The interactive command center (Dashboard) for mission prediction.
* 🖼️ `dataset_images/`: Over 9,000+ labeled frames capturing drone behavior under electronic attack.
* 📦 `drone_assets/`: Custom URDF and 3D models for the quadrotor simulation.

## 🚀 Quick Start

1. **Clone the project:**
   ```bash
git clone https://github.com/dohaziad05/-Sky-Cipher.git
