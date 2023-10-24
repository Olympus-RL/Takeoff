import torch
from torch.distributions import Uniform
from torch.nn.functional import normalize

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.torch.rotations import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz,
    quat_diff_rad,
    quat_from_euler_xyz,
)
from omni.isaac.core.utils.prims import get_prim_at_path

from Robot import Olympus, OlympusView, OlympusSpring, OlympusSpringJIT, OlympusForwardKinematics


class Jump2DTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # reward scales
        self.rew_scales = {}
        self.rew_scales["r_orient"] = self._task_cfg["env"]["learn"]["rOrientRewardScale"]
        self.rew_scales["r_base_acc"] = self._task_cfg["env"]["learn"]["rBaseAccRewardScale"]
        self.rew_scales["r_action_clip"] = self._task_cfg["env"]["learn"]["rActionClipRewardScale"]
        self.rew_scales["r_torque_clip"] = self._task_cfg["env"]["learn"]["rTorqueClipRewardScale"]
        self.rew_scales["r_jump"] = self._task_cfg["env"]["learn"]["rJumpRewardScale"]
        self.rew_scales["r_max_height"] = self._task_cfg["env"]["learn"]["rMaxHeightRewardScale"]
        self.rew_scales["total"] = self._task_cfg["env"]["learn"]["rewardScale"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = self._task_cfg["sim"]["dt"]
        self._controlFrequencyInv = self._task_cfg["env"]["controlFrequencyInv"]
        self._memory_lenght = 20
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / (self.dt * self._controlFrequencyInv) + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.max_torque = self._task_cfg["env"]["control"]["max_torque"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._olympus_translation = torch.tensor(self._task_cfg["env"]["baseInitState"]["pos"])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 27
        self._num_actions = 4
        self._num_articulated_joints = 20

        RLTask.__init__(self, name, env)
        # after rl task init we hace acess to sim device etc.

        self.lateral_motor_limits = (
            torch.tensor(self._task_cfg["env"]["jointLimits"]["lateralMotor"], device=self._device) * torch.pi / 180 * 0
        )
        self.transversal_motor_limits = (
            torch.tensor(self._task_cfg["env"]["jointLimits"]["transversalMotor"], device=self._device) * torch.pi / 180
        )

        self._min_transversal_motor_sum = (
            self._task_cfg["env"]["jointLimits"]["minTransversalMotorSum"] * torch.pi / 180
        )
        self._max_transversal_motor_sum = (
            self._task_cfg["env"]["jointLimits"]["maxTransversalMotorSum"] * torch.pi / 180
        )

        self._nominal_height = torch.tensor(pos[-1], device=self._device)
        self._target_rotation = torch.tensor([1,0,0,0], device=self._device).expand(self._num_envs, -1)
        # Random initial squat angles after reset
        init_squat_angle_limits = torch.tensor([0.0, self._max_transversal_motor_sum/2], device=self._device)
        self._init_squat_angle_sampler = Uniform(init_squat_angle_limits[0], init_squat_angle_limits[1])
        #Random initial upward velocity after reset
        init_upward_velocity_limits = torch.tensor([0.0, 0.5], device=self._device)
        self._init_upward_velocity_sampler = Uniform(init_upward_velocity_limits[0], init_upward_velocity_limits[1])
        

        self._obs_count = 0
        return

    def set_up_scene(self, scene) -> None:
        self.get_olympus()
        super().set_up_scene(scene)
        self._olympusses = OlympusView(prim_paths_expr="/World/envs/.*/Olympus/Body", name="olympusview")
        scene.add(self._olympusses)
        scene.add(self._olympusses._knees)
        scene.add(self._olympusses._base)

        scene.add(self._olympusses.MotorHousing_FL)
        scene.add(self._olympusses.FrontMotor_FL)
        scene.add(self._olympusses.BackMotor_FL)
        scene.add(self._olympusses.FrontKnee_FL)
        scene.add(self._olympusses.BackKnee_FL)
        scene.add(self._olympusses.Paw_FL)

        scene.add(self._olympusses.MotorHousing_FR)
        scene.add(self._olympusses.FrontMotor_FR)
        scene.add(self._olympusses.BackMotor_FR)
        scene.add(self._olympusses.FrontKnee_FR)
        scene.add(self._olympusses.BackKnee_FR)
        scene.add(self._olympusses.Paw_FR)

        scene.add(self._olympusses.MotorHousing_BL)
        scene.add(self._olympusses.FrontMotor_BL)
        scene.add(self._olympusses.BackMotor_BL)
        scene.add(self._olympusses.FrontKnee_BL)
        scene.add(self._olympusses.BackKnee_BL)
        scene.add(self._olympusses.Paw_BL)

        scene.add(self._olympusses.MotorHousing_BR)
        scene.add(self._olympusses.FrontMotor_BR)
        scene.add(self._olympusses.BackMotor_BR)
        scene.add(self._olympusses.FrontKnee_BR)
        scene.add(self._olympusses.BackKnee_BR)
        scene.add(self._olympusses.Paw_BR)
        return

    def get_olympus(self):
        olympus = Olympus(
            prim_path=self.default_zero_env_path + "/Olympus",
            usd_path="/Olympus-ws/Olympus-USD/Olympus/v2/olympus_v2_reorient_instanceable.usd",
            name="Olympus",
            translation=self._olympus_translation,
        )

        self._sim_config.apply_articulation_settings(
            "Olympus",
            get_prim_at_path(olympus.prim_path),
            self._sim_config.parse_actor_config("Olympus"),
        )
        # Configure joint properties
        joint_paths = []

        for quadrant in ["FR", "FL", "BR", "BL"]:
            joint_paths.append(f"Body/LateralMotor_{quadrant}")
            joint_paths.append(f"MotorHousing_{quadrant}/FrontTransversalMotor_{quadrant}")
            joint_paths.append(f"MotorHousing_{quadrant}/BackTransversalMotor_{quadrant}")
        for joint_path in joint_paths:
            set_drive(
                f"{olympus.prim_path}/{joint_path}",
                "angular",
                "position",
                0,
                self.Kp,
                self.Kd,
                self.max_torque,
            )
        self.default_articulated_joints_pos = torch.zeros(
            (self.num_envs, self._num_articulated_joints),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_actuated_joints_pos = torch.zeros(
            (self.num_envs, 12),  # self._num_actuated),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_names = olympus.dof_names
        for i in range(self._num_articulated_joints):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_articulated_joints_pos[:, i] = angle

    def get_observations(self) -> dict:

        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self._transversal_indicies)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self._transversal_indicies)
        base_velocities = self._olympusses.get_velocities(clone=False)
        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)
        height = base_position[:, -1]
        contact_states = self._olympusses.get_contact_state()
        self._flight_buf = torch.all(contact_states == 0, dim=1)
        self._max_heigth_buf = torch.max(self._max_heigth_buf, height)
        self._fallen_buf = self._olympusses.has_fallen()
        obs = torch.cat(
            (
                motor_joint_pos,
                motor_joint_vel,
                base_rotation,
                base_velocities,
                height.unsqueeze(-1),
            ),
            dim=-1,
        )

        self.obs_buf = obs.clone()

        observations = {self._olympusses.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self) -> None:
        """
        Prepares the quadroped for the next physichs step.
        NB this has to be done before each call to world.step().
        NB this method does not acceopt control signals as input,
        please see the apply_contol method.
        """

        # Check if simulation is running
        if not self._env._world.is_playing():
            return

        # calculate spring torques
        # spring_actions = self.spring.forward()
        # Handle resets
        spring_actions = self.spring.forward()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            spring_actions.joint_efforts[reset_env_ids,:] = 0.0
        self._olympusses.apply_action(spring_actions)

    def post_physics_step(self):
        """Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1
        if self._env._world.is_playing():
            self.get_observations()
            self.get_states()
            self.is_done()  # call is done to ubdate reset buffer before calculating metric in order to hand out termination rewards
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def apply_control(self, actions) -> None:
        """
        Apply control signals to the quadropeds.
        Args:
        actions: Tensor of shape (num_envs,4)
        """
        # extend the actions
        # expand points to same place in memeory. might be goofy
        extended_actions = torch.zeros((self._num_envs, 12), device=self._device)
        extended_actions[:, self._action_0_indicies] = actions[:, [0]].expand(-1, 2)
        extended_actions[:, self._action_1_indicies] = actions[:, [1]].expand(-1, 2)
        extended_actions[:, self._action_2_indicies] = actions[:, [2]].expand(-1, 2)
        extended_actions[:, self._action_3_indicies] = actions[:, [3]].expand(-1, 2)
        # lineraly interpolate between min and max
        self.current_policy_targets = (
        0.5 * extended_actions * (self.olympus_motor_joint_upper_limits - self.olympus_motor_joint_lower_limits).view(1, -1) 
        + 0.5 * (self.olympus_motor_joint_upper_limits + self.olympus_motor_joint_lower_limits).view(1, -1)
        )
        #clamp targets to avoid self collisions
        self.current_clamped_targets = self._clamp_joint_angels(self.current_policy_targets)
        # Set targets
        self._olympusses.set_joint_position_targets(self.current_clamped_targets, joint_indices=self._actuated_indicies)
    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        #sample squat angle
        squat_angles = self._init_squat_angle_sampler.rsample((num_resets,))
        k_outer, k_inner, init_heights = self._forward_kin.get_squat_configuration(squat_angles)
        #sample init vertival velocity
        vel_z = self._init_upward_velocity_sampler.rsample((num_resets,))
        #Set initial joint states
        dof_pos = self.default_articulated_joints_pos[env_ids]
        dof_pos[:, self._transversal_indicies] = squat_angles.unsqueeze(-1)
        dof_pos[:, self._knee_outer_indicies] = k_outer.unsqueeze(-1)
        dof_pos[:, self._knee_inner_indicies] = k_inner.unsqueeze(-1)
        dof_vel = torch.zeros((num_resets, self._olympusses.num_dof), device=self._device)
        root_pos = self.initial_root_pos[env_ids]
        root_pos[:, -1] = init_heights
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        root_vel[:, 2] = vel_z
        zero_rot = torch.zeros(num_resets, 4, device=self._device)
        zero_rot[:, 0] = 1.0
        # Apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._olympusses.set_joint_positions(dof_pos, indices)
        self._olympusses.set_joint_velocities(dof_vel, indices)
        self._olympusses.set_world_poses(root_pos, zero_rot, indices)
        self._olympusses.set_velocities(root_vel, indices)
        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_motor_joint_vel[env_ids] = 0.0
        self._max_heigth_buf[env_ids] = 0.0

    def post_reset(self):
        self._forward_kin = OlympusForwardKinematics(self._device)
        self.spring = OlympusSpringJIT(k=400, olympus_view=self._olympusses, equality_dist=0.2, pulley_radius=0.02)
        self.actuated_name2idx = {}
        for i, name in enumerate(self._olympusses.dof_names):
            if "Knee" not in name:
                self.actuated_name2idx[name] = i
        self._actuated_indicies = torch.tensor(list(self.actuated_name2idx.values()), device=self._device)
        self._num_actuated = len(list(self.actuated_name2idx.values()))
        # motor indicies
        self.front_transversal_indicies = torch.tensor(
            [self.actuated_name2idx[f"FrontTransversalMotor_{quad}"] for quad in ["FL", "FR", "BL", "BR"]]
        )
        self.back_transversal_indicies = torch.tensor(
            [self.actuated_name2idx[f"BackTransversalMotor_{quad}"] for quad in ["FL", "FR", "BL", "BR"]]
        )
        self.lateral_indicies = torch.tensor(
            [self.actuated_name2idx[f"LateralMotor_{quad}"] for quad in ["FL", "FR", "BL", "BR"]]
        )
        self._transversal_indicies = torch.cat((self.front_transversal_indicies, self.back_transversal_indicies))
        self._knee_inner_indicies = torch.tensor(
            [self._olympusses.get_dof_index(f"BackKnee_F{side}") for side in ["L", "R"]]
            + [self._olympusses.get_dof_index(f"FrontKnee_B{side}") for side in ["L", "R"]]
        )
        self._knee_outer_indicies = torch.tensor(
            [self._olympusses.get_dof_index(f"FrontKnee_F{side}") for side in ["L", "R"]]
            + [self._olympusses.get_dof_index(f"BackKnee_B{side}") for side in ["L", "R"]]
        )
        self._action_0_indicies = self.back_transversal_indicies[0:2]
        self._action_1_indicies = self.front_transversal_indicies[0:2]
        self._action_2_indicies = self.back_transversal_indicies[2:4]
        self._action_3_indicies = self.front_transversal_indicies[2:4]
        # joimt limits
        self.olympus_motor_joint_lower_limits = torch.zeros(
            (self._num_actuated,), device=self._device, dtype=torch.float
        )
        self.olympus_motor_joint_upper_limits = torch.zeros(
            (self._num_actuated,), device=self._device, dtype=torch.float
        )
        self.olympus_motor_joint_lower_limits[self.front_transversal_indicies] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[self.back_transversal_indicies] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[self.lateral_indicies] = self.lateral_motor_limits[0]
        self.olympus_motor_joint_upper_limits[self.front_transversal_indicies] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[self.back_transversal_indicies] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[self.lateral_indicies] = self.lateral_motor_limits[1]
        # initialize buffers
        self.initial_root_pos, self.initial_root_rot = self._olympusses.get_world_poses()
        self.current_clamped_targets = self.default_actuated_joints_pos.clone()
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_motor_joint_vel = torch.zeros(
            (self._num_envs, self._num_articulated_joints), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_vel = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.last_actions = torch.zeros(
            (self._num_envs, self.num_actions), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self._max_heigth_buf = torch.zeros_like(self.reset_buf)
        # reset all envs
        indices = torch.arange(self._olympusses.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)
        # Calculate rew_{base_acc}
        velocity = self._olympusses.get_linear_velocities(clone=False)
        rew_base_acc = -torch.norm((velocity - self.last_vel) / self.dt, dim=1) ** 2 * self.rew_scales["r_base_acc"]
        ### Calculate contionuse rewards ###
        # Calculate rew_{action_clip}
        rew_action_clip = (
            -torch.norm(self.current_policy_targets - self.current_clamped_targets, dim=1) ** 2
            * self.rew_scales["r_action_clip"]
        )
        # Calculate rew_{torque_clip}
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self._transversal_indicies)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self._transversal_indicies)
        commanded_torques = (
            self.Kp * (self.current_clamped_targets[:, self._transversal_indicies] - motor_joint_pos)
            - self.Kd * motor_joint_vel
        )
        applied_torques = commanded_torques.clamp(-self.max_torque, self.max_torque)
        rew_torque_clip = (
            -torch.norm(commanded_torques - applied_torques, dim=1) ** 2 * self.rew_scales["r_torque_clip"]
        )
        # Calculate rew_{jump}
        target_heading = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float)
        heading_error = (torch.sum(normalize(velocity) * target_heading, dim=-1).clamp(-1, 1)).acos()
        rew_jump = torch.exp(-heading_error / 0.02**2) * self.rew_scales["r_jump"]
        rew_jump[~self._flight_buf] = 0.0  # only give flight reward when flying
        # Calculate rew_{orient}
        rew_orient = -(quat_diff_rad(base_rotation,self._target_rotation))**2 * self.rew_scales["r_orient"]
        # Calculate rew_{fallen}
        rew_fallen = -self._fallen_buf.float() * 1000 #this should come from config
        # Calculate total cont reward
        total_reward_cont = (rew_base_acc + rew_action_clip + rew_torque_clip + rew_jump + rew_orient + rew_fallen) * self.rew_scales["total"]
        ### Calculate discrete rewards ###
        discrete_reward = torch.zeros_like(total_reward_cont)
        # Calculate rew_{max_height}
        max_height_dev = self._max_heigth_buf[self.reset_buf] - self._nominal_height
        rew_max_height = (max_height_dev / 0.2) ** 3 * self.rew_scales["r_max_height"]
        discrete_reward[self.reset_buf] = rew_max_height
        # Calculate total reward
        total_reward = total_reward_cont + discrete_reward
        # Save last values
        self.last_actions = self.actions.clone()
        self.last_motor_joint_vel = motor_joint_vel.clone()
        self.last_vel = velocity.clone()
        # Place total reward in buffer
        self.rew_buf = total_reward.detach().clone()
        # update extras
        self.extras["rew_base_acc"] = rew_base_acc.detach().clone()
        self.extras["rew_action_clip"] = rew_action_clip.detach().clone()
        self.extras["rew_torque_clip"] = rew_torque_clip.detach().clone()
        self.extras["rew_jump"] = rew_jump.detach().clone()
        self.extras["rew_orient"] = rew_orient.detach().clone()
        self.extras["rew_max_height"] = rew_max_height.detach().clone()
    
    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        # TODO: Collision detection
        self.reset_buf[:] = time_out#.logical_or(self._fallen_buf)

    def _clamp_joint_angels(self, joint_targets):
        clamped_targets = joint_targets.clone()
        front_pos = clamped_targets[:, self.front_transversal_indicies]
        back_pos = clamped_targets[:, self.back_transversal_indicies]

        motor_joint_sum = (front_pos + back_pos) - self._min_transversal_motor_sum
        clamp_mask_min = motor_joint_sum < 0
        front_pos[clamp_mask_min] -= motor_joint_sum[clamp_mask_min] / 2
        back_pos[clamp_mask_min] -= motor_joint_sum[clamp_mask_min] / 2

        motor_joint_sum = front_pos + back_pos - self._max_transversal_motor_sum
        clamp_mask_max = motor_joint_sum > 0
        front_pos[clamp_mask_max] -= motor_joint_sum[clamp_mask_max] / 2
        back_pos[clamp_mask_max] -= motor_joint_sum[clamp_mask_max] / 2

        clamped_targets[:, self.front_transversal_indicies] = front_pos
        clamped_targets[:, self.back_transversal_indicies] = back_pos
        return clamped_targets
