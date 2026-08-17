Setting up a new connector
==========================

Every Hoermann connector is described by data in one directory, not by Python. One
generic env, ``ur3e_hoermann_connector``, reads that directory, so adding a connector
means filling the directory in -- there is no config subclass to write and nothing to
register.

.. code-block:: bash

   tools/setup_connector.sh /media/internal/nvme/shared_data/hoermann/plugs/NewPlug

That walks through every step below and starts training at the end. The rest of this page
explains what it does, in case you need to redo a step or fix something by hand.


Before you start
----------------

Two things must already be in the connector's directory:

``mesh/``
    The 3D scan of the connector: ``mesh_textured.obj`` plus its ``.mtl`` and textures.
    The camera uses this to recognise the plug.

``object_spec.json``
    Names the mesh and tells the detector what to look for:

    .. code-block:: json

       {
         "object_name": "yellow connector",
         "mesh_path": "./mesh/mesh_textured.obj",
         "yolo_class_name": "plug_yellow_body",
         "confidence_threshold": 0.2
       }

    ``yolo_class_name`` is the one that matters -- it is what the detector actually
    searches for.

The pose publishers do the actual camera work, so they must be running for the steps that
locate the plug -- teaching the two grasp poses, and collecting (stage 5). Nothing else
needs them, and they are not needed just to start a run: the clients connect lazily, so a
missing publisher only shows up as a 30-second timeout when an estimate is first
requested.

.. code-block:: bash

   tools/run_pose_publishers_tmux.sh


What ends up in the directory
-----------------------------

=========================== ============================================ ==================
File                        What it is                                   Made by
=========================== ============================================ ==================
``mesh/``                   the 3D scan                                  you (scanner)
``object_spec.json``        mesh path + what the detector looks for      you (by hand)
``poses.json``              socket + 2 grasp positions                   stage 1
``connector.json``          LED, crops, gripper, limits, checkpoints     stages 2-4
``plugged_poses.jsonl``     where the socket really was, each insertion  stage 5
``resample_volume.json``    the socket area worked out from those        stage 6
``reward/``                 trained success classifier (optional)        stage 7
``insertion/``              training data and runs                       stage 8
=========================== ============================================ ==================


The steps
---------

**1. Teach the positions** -> ``poses.json``

All three connector-specific positions are taught in one continuous ``record.py`` run.
The manual states use compliant Cartesian force mode and SpaceMouse control.

First, the robot moves to the shared top scan pose and estimates the hanging plug. Drive
to ``grasp_pose_hanging`` and press ENTER to store the grasp relative to that estimate.
The gripper then closes automatically.

The robot closes the gripper and follows the normal pull and second-scan sequence. At the
insertion viewpoint it estimates the plug again; drive to ``grasp_pose_insert`` and press
ENTER. That pose is also stored relative to its corresponding estimate.

With the second grasp still closed, the robot moves through stretch and approach. Use the
SpaceMouse to seat the plug and press ENTER to store ``plug_pose`` in the world frame.
This final store ends the run.

The rest of the positions -- the two camera viewpoints, the approach, the pull-down and
the stretch -- are the same for every connector on this rig and are built in
(``SHARED_POSES`` in ``experiments/envs/hoermann/connector/config.py``). If one connector
genuinely needs a different value, add that pose to its ``poses.json`` and it wins over
the built-in default.

Re-teaching replaces the three taught entries and keeps any extra entries that were
added by hand.

**2. Calibrate the LED** -> ``connector.json``: ``led``

This is how the robot knows an insertion worked. Click the connector's status LED in the
live view, sample it a few times dark (``o``) and a few times lit (``n``), press ``q``.
The threshold lands halfway between.

Worth getting right: if the threshold is wrong the LED never registers, no success is
ever recorded, and a training run quietly produces nothing.

**3. Choose the crops** -> ``connector.json``: ``crop``

Click the socket in each of the two camera views so the green box sits on it. That box is
all the robot looks at while learning. ``w`` saves.

**4. Gripper opening** -> ``connector.json``: ``gripper``

Between insertions the arm opens the gripper slightly so the plug can re-seat in the
fingers, then closes and pulls out. You are asked how far to open: enter a percentage
(``70``) or a fraction (``0.7``). ``0`` is fully open and ``1`` (or ``100``) fully closed.

**5. Collect real socket positions** -> ``plugged_poses.jsonl``

The robot runs the whole cycle by itself and, on each successful insertion, notes exactly
where the socket turned out to be. It then lets go and waits for you to re-hang the cable
before going again. Aim for at least 8, ideally 15 -- fewer and the measured area comes
out far too small.

**6. Measure the socket area** -> ``resample_volume.json``

Turns those samples into a centre and a spread per axis. Everything downstream is derived
from it.

**7. Record offline insertion data** -> ``insertion/insert``

Runs the connector environment through ``record.py`` with manual SpaceMouse control and
stores insertion episodes for offline RL and reward-classifier training. Successful,
failed and near-miss attempts should all be represented. The LED supplies the success
labels. Stop after roughly 50 varied episodes, or set ``OFFLINE_EPISODES`` to another
target. Recording and training receive the same dataset parent and base repo ID.
``record.py`` writes the adaptive ``insert`` primitive to ``<root>/insert`` with repo ID
``<base>-insert``; this is exactly how ``learner_server.py`` discovers and loads the
external offline replay data.

**8. Reward classifier** (default; skipped only with ``LED_AVAILABLE=true``)

Trains a learned success detector from the newly recorded ``insertion/insert`` dataset
and uses it instead of the LED.

**9. Train**

Starts the learner and the actor in a tmux session:

.. code-block:: bash

   tmux attach -t connector_train    # Ctrl-b n switches between learner and actor
   tmux kill-session -t connector_train


Redoing one step
----------------

Before running each incomplete step, the script shows its instructions and then pauses for ENTER.
Steps are skipped if already done,
so re-running after a crash is safe. To redo one:

.. code-block:: bash

   FORCE_STAGE=led tools/setup_connector.sh /path/to/plugs/NewPlug

Stages: ``teach``, ``led``, ``crop``, ``gripper``, ``collect``, ``volume``, ``offline``,
``reward``.


``connector.json`` reference
----------------------------

Written by stages 2-4 and safe to edit by hand. Unknown keys are rejected rather than
ignored, so a typo shows up immediately.

.. code-block:: json

   {
     "schema_version": 1,
     "led": {"pixel_x": 286, "pixel_y": 323,
             "luminance_threshold": 168.0, "patch_radius": 6},
     "crop": {"params": {"wrist": [191, 310, 64, 64], "side": [195, 271, 64, 64]},
              "resize_size": [64, 64]},
     "gripper": {"reset_open_pos": 0.7},
     "insertion_bounds_half_ranges": {"x": [-1.0, 1.0], "y": [-2.0, 2.0],
                                      "z": [-1.0, 1.0], "rx": [-2.0, 2.0]},
     "reward_classifier": {"path": null, "threshold": 0.9}
   }

Policy checkpoints are deliberately not stored here. Pass one run-wide checkpoint with
``--policy.path=/path/to/pretrained_model``; omit it to use the environment policy config.

``led``
    Required unless ``--env.use_reward_classifier=true``. Crop boxes are
    ``[top, left, height, width]``: ``top`` is the y coordinate, ``left`` is x.

``crop.params``
    Set to ``null`` (with ``resize_size`` also ``null``) to train on full-resolution
    images.

``gripper.reset_open_pos``
    Accepts a fraction (``0.7``) or a percentage (``70``).

``insertion_bounds_half_ranges``
    How far the policy may move, per axis, in multiples of the spread measured in stage 6.
    The defaults are symmetric: ``x`` uses 1, ``y`` uses 2, ``z`` uses 1, and ``rx`` uses 2
    measured half-ranges in either direction. Override any connector with explicit
    ``[low_multiplier, high_multiplier]`` values here; asymmetric values remain supported.
    Because these are multipliers rather than distances, they stay meaningful after the
    socket area is re-measured. ``ry``/``rz`` are not policy axes and are always unbounded.

``reward_classifier.path``
    Required only when running with ``--env.use_reward_classifier=true``.


Running it by hand
------------------

.. code-block:: bash

   python src/share/scripts/record.py \
       --env.type=ur3e_hoermann_connector \
       --env.object_dir=/media/internal/nvme/shared_data/hoermann/plugs/NewPlug

The generic defaults use the conservative manual-evaluation profile: full reset, manual
success signaling, no seating push, an unfrozen vision encoder, and wrist-camera-only
policy observations. ``setup_connector.sh`` overrides success detection after it has
calibrated an LED or trained a reward classifier.

Useful switches: ``--env.collect_plug_poses=true`` (stage 5),
``--env.use_reward_classifier=true`` (learned success detector instead of the LED),
``--env.skip_pose_estimation=true`` (start at the insertion, skipping the pick).

Settings that describe the *connector* -- LED, crops, gripper, limits -- deliberately have
no command-line flags. They live in ``connector.json`` so that whatever a run used is
recorded next to the connector rather than in someone's shell history.


If something goes wrong
-----------------------

**The LED never registers a success.** Re-run stage 2 (``FORCE_STAGE=led``). Check the
lighting has not changed since it was calibrated.

**The robot cannot find the socket / insertion always fails.** Check ``n_samples`` in
``resample_volume.json``. If it is small the measured area is too tight; collect more with
``FORCE_STAGE=collect`` and re-run stage 6.

**The camera cannot find the plug.** Check the pose publishers are running
(``tmux has-session -t pose_publishers``) and that ``yolo_class_name`` in
``object_spec.json`` is right.

**"connector.json is missing ..."** The message names the stage that writes it; re-run
that stage.


Why the e2e configs still exist
-------------------------------

``experiments/envs/hoermann/e2e/*`` are frozen per-connector snapshots from before this
was data-driven. ``ur3e_hoermann_connector`` builds a graph identical to
``RightTTL_090726`` when given the same data -- asserted in
``tests/experiments/envs/test_connector_config.py`` -- and the e2e configs are left
untouched so old runs stay reproducible. New connectors should not copy them.
