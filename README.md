# Fabrics

Fabrics are nonlinear, autonomous, second order differential equations that are provably stable and exhibit path consistency properties. This library brings fabrics to the GPU enabling large-scale parallelization, differentiability, and more. The intended domain is robot control and we include a manually designed fabric policy for the Kuka-Allegro robot that enabled DextrAH training.

![](./docs/img/gifs/fabrics_collage.gif)

## Installation
**Note**: This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

1. [Install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) Isaac Sim following the local conda install route.

2. Install Fabrics within your new conda env
```bash
        curl -sSL https://install.python-poetry.org | python3 - --version 1.8.3
        git lfs clone git@github.com:NVlabs/FABRICS.git
        cd <FABRICS>
        poetry install
        or
        python -m pip install -e .
```
3. Patch `urdfpy` dependency `networkx`
```bash
       cd <fabrics_dir>
       chmod +x urdfpy_patch.sh
       ./urdfpy_patch.sh
```
4. Ensure high enough `GLIBCXX_` can be found
```bash
        conda install -c conda-forge libstdcxx-ng
        conda install -c conda-forge libgcc-ng=12 libstdcxx-ng=12
```

## Franka Panda pose fabric (added in this fork)

`FrankaPoseFabric` (`src/fabrics_sim/fabrics/franka_pose_fabric.py`) drives the 7 arm joints
of a Franka Panda toward a 6-DoF **tool-centre-point pose** target while respecting joint,
acceleration and jerk limits, self-collision and world collision. The gripper is not part of
the fabric (finger joints are fixed open in the fabric URDF; command the gripper in the sim).

```bash
# headless check: random vertical-gripper targets over a table, then a below-table target
python examples/franka_pose_fabric_example.py --cuda_graph
python tests/test_franka_fk.py              # fabric FK == urdfpy FK
python tests/test_franka_home_clearance.py  # no sphere penetrates the world at home
```

Assets are generated, not hand-written: `tools/make_franka_fabric_assets.py` turns the stock
`panda_arm_hand.urdf` into `models/robots/urdf/franka/franka.urdf` (fixed fingers, `panda_tcp`
frame 0.1034 m past `panda_hand`, six +-0.25 m axis-point frames, and k-means-fitted collision
sphere frames), writes `fabric_params/franka_pose_params.yaml` and `worlds/franka_table.yaml`.

Verified (RTX 3080, batch 8, 60 Hz, 6 s per target, 48 holds per run): 100% of targets within
1 cm / 5 deg (median 0.1 mm / 0.0 deg) across the box x 0.30-0.65, y +-0.35, z 0.03-0.50 with
random yaw; no collision; a target 15 cm below the table stops at the table surface. With
`--cuda_graph`, 6 s of control run in ~0.35 s wall.

Worlds: `worlds/franka_table.yaml` (table + static base box) and `worlds/franka_cylinders.yaml`
(adds three pillars of different heights for collision-avoidance tests). Collision-avoidance
accelerations default to forcing 5 / geometric 15; on the pillar course the Kuka defaults (1 / 5)
penetrated the sphere model by up to 0.4 cm, the new values keep it non-negative.

Notes:
* The posture attractor's home is a mid-workspace *reach* posture, not the folded IsaacLab
  start posture; with the folded home the wrist over-pitched into its limit on low targets.
* The IsaacLab Franka start posture holds the hand 45.6 deg from vertical. Pass exactly vertical
  orientations (180 deg about world x, plus yaw) for pick-and-place.
* Fixes to the library needed by this fabric: `quaternion_to_matrix_kernel` used a
  `wp.quaternion` constructor that does not exist in warp 1.8 (now `wp.quat`), and quaternion
  reordering uses slices so `set_features` is CUDA-graph capturable.

## Examples
Once installed, you should be able to run the example script in the example directory, e.g.:

Kuka-Allegro (DextrAH) fabric:

    python <fabrics_dir>/examples/kuka_allegro_pose_fabric_example.py --batch_size=10 --render --cuda_graph

## Notes
One can update dependences in deps.txt file, remove pyproject.toml and poetry.lock files, and regenerate them
```bash
    cd <fabrics_dir>
    rm pyproject.toml poetry.lock
    poetry init --name "fabrics_sim" --no-interaction
    xargs poetry add < deps.txt
```
Add the following after ```authors``` in pyproject.toml
```bash
    packages = [ 
    { include = "fabrics_sim", from = "src" }
    ]
```
Install the project
```bash
    poetry install
```
