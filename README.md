# MuJoCo tutorial environments

This repo contains a small set of MuJoCo XML scenes plus Python scripts that visualize
or control them. Use the XML files as "envs" and the scripts as examples of running
or controlling each scene.

## Environments (XML)

- `hello.xml`
  - Minimal scene: ground plane, light, and a free box dropped above the plane.
- `example.xml`
  - Simple articulated arm (ball joint + two hinges) plus a free cylinder.
  - Includes a spatial tendon between two sites.
- `example_with_actuators.xml`
  - Same arm scene as `example.xml`, with named joints and motors added.
  - Useful for joint-space control demos.
- `quadcopter.xml`
  - Quadrotor body with four rotor sites and general actuators.
  - Includes gravity, RK4 integrator, and a ground plane.
- `robot_hand.xml`
  - Multi-finger hand model with hinge joints and per-joint motors.
  - Higher-frequency timestep and tuned joint friction/damping.

## Scripts

- `visualize.py`  
  Visualize any XML (defaults to `hello.xml`).
- `visualize_hand.py`  
  Visualize the hand with a pre-framed camera (defaults to `robot_hand.xml`).
- `control_actuators.py`  
  PD control to hold arm joints at their initial angles
  (defaults to `example_with_actuators.xml`).
- `arm_curl.py`  
  PD-driven curling motion for two arm hinges
  (defaults to `example_with_actuators.xml`).
- `quadcopter_square.py`  
  Simple quadcopter controller that flies a smooth looping path
  (defaults to `quadcopter.xml`).
- `fist_open.py`  
  PD control to close the hand into a fist and open it back up
  (defaults to `robot_hand.xml`).

## Quick start

```bash
# install dependencies
python -m pip install mujoco

# visualize the simplest scene
python visualize.py

# visualize a specific env
python visualize.py quadcopter.xml

# arm control demos
python control_actuators.py
python arm_curl.py --period 3.0 --amplitude 1.0

# quadcopter loop
python quadcopter_square.py --radius 1.5 --height 1.2 --yaw-follow

# hand demo
python visualize_hand.py
python fist_open.py --period 5.0
```

## Install & setup

### 1) Install MuJoCo + Python bindings

```bash
python -m pip install mujoco
```

If you use a virtual environment, activate it before installing.

### 2) Run a demo

```bash
python visualize.py
```

### 3) macOS users

MuJoCo's viewer requires `mjpython` on macOS. Example:

```bash
mjpython visualize.py
```

## macOS note

On macOS, MuJoCo's viewer requires running under `mjpython`. The scripts will raise a
helpful error if they detect the wrong interpreter. Example:

```bash
mjpython visualize.py
mjpython fist_open.py
```
