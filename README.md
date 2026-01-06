# 🌡️ Shimoda Dormitory Thermal Simulator

**A Group Assignment for [Course Name] - SFC & Yagami Collaboration**

<img width="3144" height="1828" alt="image" src="https://github.com/user-attachments/assets/689fb99f-46ac-4782-aca9-73611aa2c8d5" />


## 📖 Overview
This project is an interactive numerical simulation tool designed to model **Heat Diffusion** in large indoor spaces. 

Developed as a collaboration between **SFC** (Policy & Environment) and **Yagami** (Science & Engineering) students, the tool specifically models the common study room of the Shimoda Dormitory. It allows users to test different heater placements and room configurations to optimize thermal comfort and energy efficiency.

## ✨ Key Features
* **Interactive Room Editor:** Draw custom room shapes, walls, and obstacles using a grid interface.
* **Real-Time Simulation:** Solves the 2D Heat Equation using the Finite Difference Method (FDM).
* **Customizable Physics:** Configure heater positions, window locations, and thermal loss factors.
* **Analytics Dashboard:** * **Heat Map:** Visualizes temperature distribution ($T(x,y)$).
    * **Efficiency Graph:** Tracks average room temperature over time.
    * **Time-to-Comfort:** Automatically calculates minutes required to reach $20^\circ C$.

## 🧮 The Math Behind It
The simulation is governed by the parabolic **Partial Differential Equation (PDE)** known as the Heat Equation:

$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u + S
$$

To solve this computationally, we utilize the **Finite Difference Method (FDM)** with a Forward Euler scheme:

$$
u_{i,j}^{n+1} = u_{i,j}^n + \frac{\alpha \Delta t}{\Delta x^2} \left( u_{i+1,j}^n + u_{i-1,j}^n + u_{i,j+1}^n + u_{i,j-1}^n - 4u_{i,j}^n \right)
$$

### Boundary Conditions
* **Heaters:** Modeled as Dirichlet boundary conditions (fixed high temperature proxy for power output).
* **Walls & Windows:** Modeled using a "Loss Factor" approach to simulate thermal mass and dispersion to the external environment ($10^\circ C$).

## 🚀 Installation & Usage

### Prerequisites
You need Python installed. We recommend creating a virtual environment.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/thermal-simulator.git](https://github.com/your-username/thermal-simulator.git)
cd thermal-simulator
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Application
```bash
streamlit run app.py
```
