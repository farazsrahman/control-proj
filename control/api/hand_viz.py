"""Geometry + rollout dump for the offline hand renderer (the Isaac Sim / USD side; the renderer itself is `utils/hand_render.py`).

    from api import hand_viz
    meshes = hand_viz.export_meshes(env)                      # link-frame visual meshes of the hand and the object
    data = hand_viz.record_rollout(env, policy, n=288, backlash=True, steps=200)
    fk = hand_viz.link_poses(env, q)                          # link poses at ARBITRARY joint positions, by the sim's own kinematics

`link_poses` writes the joint positions into the articulation, refreshes PhysX kinematics (`sim.forward()`) and reads the link poses back, so the
offline renderer needs no URDF / joint-axis bookkeeping and cannot disagree with the simulator's kinematics. It destroys the env state: call it
after the rollouts. All poses are in each env's local frame (world minus `scene.env_origins`), quaternions (w, x, y, z).
"""

from __future__ import annotations

import numpy as np
import torch


def _tri(counts, idx):
    """Fan-triangulate a USD polygon mesh."""
    tris, k = [], 0
    for c in counts:
        for j in range(1, c - 1):
            tris.append((idx[k], idx[k + j], idx[k + j + 1]))
        k += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def export_meshes(env, env_idx: int = 0, verbose: bool = True):
    """{body name: (verts [V,3], faces [F,3])} of the VISUAL meshes in each link's own frame, for the hand's links and for the object.

    Read from the USD stage the sim loaded (instance proxies included). Each mesh is mapped into its link frame by the stage's own prim
    transforms, (mesh -> world) then (world -> link), so the result does not depend on the pose PhysX has since moved the links to.
    Collision-only meshes are skipped. Needs a running app (api.isaac.launch) and the env.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    u = env.unwrapped
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out, skipped = {}, {}
    for kind, path, names in [("hand", f"/World/envs/env_{env_idx}/Robot", set(u.hand.body_names)),
                              ("object", f"/World/envs/env_{env_idx}/object", {"object"})]:
        root = stage.GetPrimAtPath(path)
        assert root.IsValid(), f"no prim at {path}"
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            p, link = prim, None
            while p.IsValid() and p.GetPath() != root.GetPath().GetParentPath():
                if p.GetName() in names:
                    link = p
                    break
                p = p.GetParent()
            if link is None:
                skipped["no-link"] = skipped.get("no-link", 0) + 1
                continue
            sp = prim.GetPath().pathString.lower()
            purpose = UsdGeom.Imageable(prim).ComputePurpose()
            if "collision" in sp or purpose == UsdGeom.Tokens.guide:
                skipped["collision"] = skipped.get("collision", 0) + 1
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts, idx = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
            if pts.size == 0 or counts is None:
                continue
            faces = _tri(counts, idx)
            m = np.array(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(link).GetInverse())  # row-vector convention
            verts = pts @ m[:3, :3] + m[3, :3]
            name = link.GetName()
            if name in out:
                v0, f0 = out[name]
                out[name] = (np.concatenate([v0, verts]), np.concatenate([f0, faces + len(v0)]))
            else:
                out[name] = (verts, faces)
    if verbose:
        print(f"export_meshes: {len(out)} bodies with visuals, {sum(len(v) for v, _ in out.values())} verts, "
              f"{sum(len(f) for _, f in out.values())} tris; skipped {skipped}; hand bodies without a mesh: "
              f"{[b for b in u.hand.body_names if b not in out]}")
    return {k: (v.astype(np.float32), f.astype(np.int32)) for k, (v, f) in out.items()}


def _state(env):
    u = env.unwrapped
    o = u.scene.env_origins
    h = u.hand.data
    return dict(q=h.joint_pos.clone(), pos=(h.body_pos_w - o[:, None]).clone(), quat=h.body_quat_w.clone(),
                obj_pos=(u.object.data.root_pos_w - o).clone(), obj_quat=u.object.data.root_quat_w.clone(),
                ep=u.episode_length_buf.clone(), goal_quat=u.goal_rot.clone(), succ=u.successes.clone())


@torch.no_grad()
def link_poses(env, q: torch.Tensor):
    """Link poses [E, B, 3], [E, B, 4] (env-local) with the hand at joint positions `q` [E, nd] (one row per env), by the sim's own FK."""
    u = env.unwrapped
    hand = u.hand
    n = u.num_envs
    assert q.shape[0] <= n
    qq = torch.zeros(n, q.shape[1], device=q.device)
    qq[: q.shape[0]] = q
    hand.write_joint_state_to_sim(qq, torch.zeros_like(qq))
    u.sim.forward()
    hand.update(1.0)  # bumps the data timestamp so the cached body states are re-read (dt only matters for finite-difference velocities)
    o = u.scene.env_origins
    return (hand.data.body_pos_w - o[:, None])[: q.shape[0]].clone(), hand.data.body_quat_w[: q.shape[0]].clone()


@torch.no_grad()
def record_rollout(env, policy, n: float = 0.0, backlash: bool = True, steps: int = 200, seed: int = 1, n_rec: int = 4):
    """Greedy rollout under measurement error (backlash only by default; n = gear ratio, 0 = clean) recording, for the first `n_rec` envs,
    the TRUE joint positions, the MEASURED ones the actor was fed (captured from MeasurementModel.apply), the sim's link poses, the cube and the goal orientation (`goal_quat`, `succ` = goals reached so far).
    Returns {name: np.ndarray [T, n_rec, ...]}.
    """
    u = env.unwrapped
    meas = u._meas
    orig, last = meas.apply, {}

    def wrapped():
        out = orig()
        last["q_true"], last["q_meas"] = u.hand_dof_pos.clone(), out[0].clone()
        return out

    meas.apply = wrapped
    u.set_measurement(n=n, backlash=backlash)
    try:
        obs, _ = env.reset(seed=seed)
        q_reset = _state(env)["q"][:n_rec].float().cpu().numpy()  # the state of the first observation: where the play operator starts
        rec = {k: [] for k in ["q_true", "q_meas", "pos", "quat", "obj_pos", "obj_quat", "ep", "goal_quat", "succ"]}
        for _ in range(steps):
            obs, *_ = env.step(policy(obs))
            s = _state(env)
            qt = last.get("q_true", s["q"])
            qm = last.get("q_meas", s["q"]) if u._meas.active else s["q"]
            for k, v in dict(q_true=qt, q_meas=qm, **{k: s[k] for k in ["pos", "quat", "obj_pos", "obj_quat", "ep", "goal_quat", "succ"]}).items():
                rec[k].append(v[:n_rec].float().cpu().numpy())
    finally:
        meas.apply = orig
        u.set_measurement(n=0)
    return {**{k: np.stack(v) for k, v in rec.items()}, "q_reset": q_reset}


@torch.no_grad()
def drive_joint(env, joint: str, waypoints_deg, frames_per_seg: int = 60, n: float = 200.0, batch: int | None = None):
    """Move ONE joint through piecewise-linear `waypoints_deg` (every other joint at its default) and return, for a single 'env' (axis 1 of size 1):
    the true joint positions, the positions the backlash model reports (the sim's own `MeasurementModel`, fed this trajectory, so it is the exact
    implementation and not a copy), and the link poses at both (`fk_true_*`, `fk_meas_*`, by the sim's kinematics). Destroys the env state.
    Needs `env.reset()` to have run (MeasurementModel reads the env's cached fingertip state).
    """
    u = env.unwrapped
    j = u.hand.joint_names.index(joint)
    base = u.hand.data.default_joint_pos[0].clone()
    wp = np.deg2rad(np.asarray(waypoints_deg, dtype=np.float64))
    traj = np.concatenate([np.linspace(a, b, frames_per_seg, endpoint=False) for a, b in zip(wp[:-1], wp[1:])] + [wp[-1:]])
    T = len(traj)
    q = base[None].repeat(T, 1)
    q[:, j] = torch.as_tensor(traj, dtype=q.dtype, device=q.device)

    u.set_measurement(n=n, backlash=True)  # resets the operator: it starts centred on the first frame
    qm = torch.empty_like(q)
    for t in range(T):
        u.hand_dof_pos = q[t][None].repeat(u.num_envs, 1)
        u.hand_dof_vel = torch.zeros_like(u.hand_dof_pos)
        qm[t] = u._meas.apply()[0][0]
    u.set_measurement(n=0)

    batch = batch or u.num_envs
    out = {"q_true": q.cpu().numpy()[:, None], "q_meas": qm.cpu().numpy()[:, None], "joint_index": np.array(j), "n": np.array(n),
           "limits": u.hand.data.soft_joint_pos_limits[0, j].cpu().numpy()}
    for key, qq in [("true", q), ("meas", qm)]:
        P, Q = [], []
        for i in range(0, T, batch):
            p, qu = link_poses(env, qq[i : i + batch])
            P.append(p.cpu().numpy())
            Q.append(qu.cpu().numpy())
        out[f"fk_{key}_pos"] = np.concatenate(P)[:, None]
        out[f"fk_{key}_quat"] = np.concatenate(Q)[:, None]
    return out
