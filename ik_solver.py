"""Standalone Pinocchio-based IK solver for Piper 6-DOF arm.

No ROS dependencies.

Two modes:
  fast    – Jacobian pseudoinverse iterative IK, <1 ms per call (for RL training)
  precise – CasADi + IPOPT optimization IK, ~20-100 ms per call (for deployment)
"""
import os
import numpy as np
import pinocchio as pin

_DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_assets", "piper_description.urdf",
)
_DEFAULT_MESH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_assets",
)


class IKSolver:
    """Dual-mode IK solver for the Piper 6-DOF arm (joints 1-6, gripper locked).

    Coordinate convention
    ---------------------
    All positions / quaternions are expressed in the arm's URDF root frame
    (the same frame as the ``base_arm_link`` body in the MuJoCo model).

    To transform a world-frame target before calling ``solve``::

        arm_pos = data.body("base_arm_link").xpos          # (3,)
        arm_rot = data.body("base_arm_link").xmat.reshape(3, 3)
        target_arm = arm_rot.T @ (target_world - arm_pos)

    Quaternion order
    ----------------
    All quaternion arguments / return values use the **[w, x, y, z]** convention.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        mesh_dir: str | None = None,
        mode: str = "fast",
    ):
        """
        Parameters
        ----------
        urdf_path : path to ``piper_description.urdf`` (default: model_assets/)
        mesh_dir  : directory containing the ``piper_description`` package tree,
                    i.e. the parent of the ``piper_meshes/`` folder.
                    Used to resolve ``package://piper_description/meshes/…``
                    URIs inside the URDF.
        mode      : ``'fast'`` or ``'precise'``
        """
        if urdf_path is None:
            urdf_path = _DEFAULT_URDF
        if mesh_dir is None:
            mesh_dir = _DEFAULT_MESH_DIR

        self.mode = mode
        self._urdf_path = urdf_path
        self._mesh_dir = mesh_dir
        self._package_dirs = [mesh_dir]

        self._init_robot()
        if mode == "precise":
            self._init_casadi()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_robot(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # Build full robot (includes gripper joints 7 & 8)
        self._full_robot = pin.RobotWrapper.BuildFromURDF(
            self._urdf_path,
            package_dirs=self._package_dirs,
        )

        # Reduced model: lock gripper joints so we solve for 6 DOF only
        # reference_configuration omitted — defaults to pin.neutral() which
        # avoids numpy ABI mismatch when pinocchio and other libraries load
        # different numpy C extensions.
        self._reduced = self._full_robot.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
        )

        # Add a named EE frame at joint6 tip (identity offset matches URDF ee)
        self._reduced.model.addFrame(
            pin.Frame(
                "ee",
                self._reduced.model.getJointId("joint6"),
                pin.SE3.Identity(),
                pin.FrameType.OP_FRAME,
            )
        )
        # Rebuild data after adding frame
        self._reduced.data = self._reduced.model.createData()
        self._ee_id = self._reduced.model.getFrameId("ee")

        self.nq = self._reduced.model.nq
        self.q_lower = self._reduced.model.lowerPositionLimit.copy()
        self.q_upper = self._reduced.model.upperPositionLimit.copy()

        # Geometry model (needed for collision checking in precise mode)
        self._geom_model = pin.buildGeomFromUrdf(
            self._full_robot.model,
            self._urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=self._package_dirs,
        )
        for i in range(4, 10):
            for j in range(0, 3):
                self._geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self._geom_data = pin.GeometryData(self._geom_model)

    def _init_casadi(self):
        import casadi
        from pinocchio import casadi as cpin  # type: ignore

        self._casadi = casadi
        cmodel = cpin.Model(self._reduced.model)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", self.nq, 1)
        cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(cmodel, cdata, cq)

        error_fn = casadi.Function(
            "error",
            [cq, cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        cdata.oMf[self._ee_id].inverse() * cpin.SE3(cTf)
                    ).vector
                )
            ],
        )

        opti = casadi.Opti()
        var_q = opti.variable(self.nq)
        param_tf = opti.parameter(4, 4)
        err = error_fn(var_q, param_tf)
        cost = casadi.sumsqr(err[:3]) + 0.1 * casadi.sumsqr(err[3:])
        reg = casadi.sumsqr(var_q)
        opti.minimize(20.0 * cost + 0.01 * reg)
        opti.subject_to(opti.bounded(self.q_lower, var_q, self.q_upper))
        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-4},
            "print_time": False,
        }
        opti.solver("ipopt", opts)

        self._opti = opti
        self._var_q = var_q
        self._param_tf = param_tf
        self._ipopt_init = pin.neutral(self._reduced.model).copy()

    # ------------------------------------------------------------------
    # Numpy ABI helpers — pinocchio's C++ needs arrays from its own numpy
    # ------------------------------------------------------------------

    def _to_pin_q(self, q_like, clip=False):
        """Convert any array-like to a pinocchio-compatible (nq,) vector."""
        q = pin.neutral(self._reduced.model).copy()
        vals = q_like.tolist() if hasattr(q_like, 'tolist') else list(q_like)
        lo = self.q_lower
        hi = self.q_upper
        for i, v in enumerate(vals):
            v = float(v)
            if clip:
                v = max(float(lo[i]), min(float(hi[i]), v))
            q[i] = v
        return q

    @staticmethod
    def _pin_vec3(xyz):
        """Create a pinocchio-compatible (3,) translation vector."""
        v = pin.SE3.Identity().translation.copy()
        for i in range(3):
            v[i] = float(xyz[i])
        return v

    def _clip_q(self, q):
        """Clip joint values in-place (pinocchio-safe, no np.clip)."""
        for i in range(self.nq):
            q[i] = max(float(self.q_lower[i]), min(float(self.q_upper[i]), float(q[i])))
        return q

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa manipulability index: sqrt(det(J @ J.T)).

        Measures how far the current configuration is from kinematic
        singularity.  Higher values = better dexterity.

        Parameters
        ----------
        q : joint angles (6,) in radians

        Returns
        -------
        Manipulability scalar (>= 0).
        """
        model = self._reduced.model
        data = self._reduced.data
        q_pin = self._to_pin_q(q, clip=True)
        pin.framesForwardKinematics(model, data, q_pin)
        pin.computeJointJacobians(model, data, q_pin)
        J = pin.getFrameJacobian(model, data, self._ee_id, pin.ReferenceFrame.LOCAL)
        det_val = np.linalg.det(J @ J.T)
        return float(np.sqrt(max(0.0, det_val)))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics.

        Parameters
        ----------
        q : joint angles (6,) in radians

        Returns
        -------
        pos  : EE position in arm frame  (3,)
        quat : EE orientation [w, x, y, z]  (4,)
        """
        q_pin = self._to_pin_q(q, clip=True)
        pin.framesForwardKinematics(
            self._reduced.model, self._reduced.data, q_pin
        )
        frame = self._reduced.data.oMf[self._ee_id]
        pos = frame.translation.copy()
        q_rot = pin.Quaternion(frame.rotation)  # internal xyzw
        quat = np.array([q_rot.w, q_rot.x, q_rot.y, q_rot.z], dtype=float)
        return pos, quat

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_joints: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Solve IK.

        Parameters
        ----------
        target_pos    : desired EE position in arm frame  [x, y, z]
        target_quat   : desired EE orientation  [w, x, y, z]
        current_joints: current 6-DOF joint angles in radians

        Returns
        -------
        (q_sol, success) :
            q_sol is np.ndarray(6,) on success, None on failure.
            success is True when IK converged to within tolerance.
        """
        # Build target SE3 — use Python floats to avoid numpy ABI mismatch
        tq = [float(x) for x in target_quat]
        tp = [float(x) for x in target_pos]
        quat = pin.Quaternion(tq[0], tq[1], tq[2], tq[3])
        target_se3 = pin.SE3(quat.toRotationMatrix(),
                             self._pin_vec3(tp))

        q0 = self._to_pin_q(current_joints, clip=True)

        if self.mode == "fast":
            return self._solve_fast(target_se3, q0)
        return self._solve_precise(target_se3, q0)

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _solve_fast(
        self,
        target_se3: pin.SE3,
        q0: np.ndarray,
        n_iter: int = 200,
        eps: float = 1e-4,
        dt: float = 0.5,
        damp: float = 1e-6,
    ) -> tuple[np.ndarray | None, bool]:
        """Jacobian pseudoinverse iterative IK (damped least-squares)."""
        model = self._reduced.model
        data = self._reduced.data
        q = q0.copy()

        for _ in range(n_iter):
            pin.framesForwardKinematics(model, data, q)
            # Error in local frame: log(T_ee^-1 * T_target)
            iMd = data.oMf[self._ee_id].actInv(target_se3)
            err = pin.log6(iMd).vector
            if np.linalg.norm(err) < eps:
                return q.copy(), True

            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(model, data, self._ee_id, pin.ReferenceFrame.LOCAL)
            # Damped pseudoinverse: J^T (JJ^T + λI)^{-1}
            JJt = J @ J.T + damp * np.eye(6)
            dq = dt * J.T @ np.linalg.solve(JJt, err)
            q = pin.integrate(model, q, dq)
            self._clip_q(q)

        # Return best-effort (caller decides based on bool)
        err_norm = np.linalg.norm(err)  # noqa: F821 (defined in loop)
        return q.copy(), err_norm < 1e-2

    def _solve_precise(
        self, target_se3: pin.SE3, q0: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        """CasADi IPOPT IK with self-collision checking."""
        self._opti.set_initial(self._var_q, q0)
        self._opti.set_value(self._param_tf, target_se3.homogeneous)
        try:
            self._opti.solve_limited()
            q_sol = self._opti.value(self._var_q)
            self._ipopt_init = q_sol.copy()
            collision = self._check_collision(q_sol)
            return q_sol.copy(), not collision
        except Exception:
            return None, False

    def _check_collision(self, q: np.ndarray) -> bool:
        q_full = pin.neutral(self._full_robot.model)
        for i in range(len(q)):
            q_full[i] = float(q[i])
        pin.forwardKinematics(
            self._full_robot.model, self._full_robot.data, q_full
        )
        pin.updateGeometryPlacements(
            self._full_robot.model,
            self._full_robot.data,
            self._geom_model,
            self._geom_data,
        )
        return pin.computeCollisions(self._geom_model, self._geom_data, False)
