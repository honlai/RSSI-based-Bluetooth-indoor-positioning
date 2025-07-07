# RSSI-based Bluetooth Indoor Positioning

This project estimates the indoor location of a device using RSSI (Received Signal Strength Indicator) values from multiple Bluetooth receivers.

* [中文說明 (Chinese README)](README.zh.md) *(Under construction)*
* 
## Goal

Given a sequence of RSSI data from multiple receivers, estimate the 2D position of a (temporarily stationary) device at time step $t$ as $\hat{\mathbf{x}}_t = [x_t, y_t]$, such that the predicted position is as close as possible to the true position.

> The original problem is in 3D: $[x_t, y_t, z_t]$, but we simplify it to 2D in this version.

## Input

At time $t$, the system receives RSSI values from $N$ Bluetooth receivers:

$$
\mathbf{z}_t = \{ r_t^{(1)}, r_t^{(2)}, ..., r_t^{(N)} \}
$$

## Output

The estimated position at time $t$:

$$
\hat{\mathbf{x}}_t = [x_t, y_t]
$$

## Requirements

- Python 3.x
- `numpy`, `pandas`, `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

##  Usage

Estimate position from the RSSI input:

```bash
python src/main.py
```

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

##  TODO

- [ ] Add support for real-time RSSI scanning using Bluetooth adapter
- [ ] Implement RSSI-to-distance model (e.g., log-distance path loss)
- [ ] Develop 2D position estimation algorithm (e.g., trilateration)
- [ ] Build visualization tools to display estimated positions
- [ ] Create dataset format and save/load utilities
- [ ] Write unit tests for core functionss
- [ ] Refactor code for modularity and reusability
- [ ] Benchmark accuracy with simulated and real data
- [ ] Add configuration file for receiver layout and parameters
- [ ] Add Chinese README (`README.zh.md`)

