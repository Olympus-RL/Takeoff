from typing import Tuple
import torch
from torch import Tensor
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
from omni.isaac.core.utils.stage import get_current_stage


from Robot import Olympus, OlympusView,OlympusSpringJIT, OlympusForwardKinematics
from loggers import JumpLogger


class LandingTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._is_test = self._cfg["test"]
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
        self._step_dt = self.dt * self._controlFrequencyInv
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self._max_steps_after_take_off = int(0.2/self._step_dt) # reste immiduatly after takeoff
        self.max_episode_length = int(self.max_episode_length_s / (self.dt * self._controlFrequencyInv) + 0.5)

        #actuators
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.max_torque = self._task_cfg["env"]["control"]["max_torque"]
        self._motor_cutoff_speed = 350/30*torch.pi #self._task_cfg["env"]["control"]["motorCutoffFreq"]
        self._motor_degration_speed = 225/30*torch.pi #self._task_cfg["env"]["control"]["motorDegradationFreq"]
        self._ema_cut_off_freq = torch.tensor(5.0)#hz
        self._ema_alpha = 1 - torch.exp(-1/(self._ema_cut_off_freq*self._step_dt))
        self._ema_alpha = torch.tensor(0.9)
        print("ema alpha", self._ema_alpha)

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._olympus_translation = torch.tensor(self._task_cfg["env"]["baseInitState"]["pos"])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 40
        self._num_actions = 12
        self._num_articulated_joints = 20
    
        RLTask.__init__(self, name, env)
        # after rl task init we hace acess to sim device etc.

        self.lateral_motor_limits = (
            torch.tensor(self._task_cfg["env"]["jointLimits"]["lateralMotor"], device=self._device) * torch.pi / 180 
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
        #Random stuff after reste
        init_squat_angle_limits = torch.tensor([torch.pi/2, self._max_transversal_motor_sum/2], device=self._device)
        self._init_squat_angle_sampler = Uniform(init_squat_angle_limits[0], init_squat_angle_limits[1])
        init_upward_velocity_limits = torch.tensor([0.0, 0.5], device=self._device)
        self._init_upward_velocity_sampler = Uniform(init_upward_velocity_limits[0], init_upward_velocity_limits[1])

        

        # curiculum
        self._curriculum_init_squat_angle_lower = torch.tensor([90.0,5.1,5.1], device=self._device).deg2rad()
        self._curriculum_init_squat_angle_upper = torch.tensor([120.0,10.0,10.0], device=self._device).deg2rad()
        self._curriculum_init_base_height_lower = torch.tensor([0.0,0,0.0], device=self._device)
        self._curriculum_init_base_heigt_upper = torch.tensor([0.0,0,0.0], device=self._device)
        self._curriculum_tresh = 0.05
        self._n_curriculum_levels = 3
        self._steps_per_curriculum_level = 5

        self._obs_count = 0

        self._ema_alpha.to(self._device)

        self._zero_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).expand(self._num_envs, -1)

        return

    def set_up_scene(self, scene) -> None:

        self.get_olympus()
        super().set_up_scene(scene)
        self._olympusses = OlympusView(prim_paths_expr="/World/envs/.*/Olympus/Body", name="olympusview")
        scene.add(self._olympusses)


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
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self._actuated_indicies)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self._actuated_indicies)
        base_velocities = self._olympusses.get_velocities(clone=False)
        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)
        height = base_position[:, -1]
        self._contact_states, self._collision_buf = self._olympusses.get_contact_state_collisionbuf()
        self._collision_buf = self._collision_buf.logical_or(height < 0.175)
        self._paw_height = self._olympusses.get_paw_heights()
        
        new_obs = torch.cat(
            (   
                self._target_height.unsqueeze(-1),
                motor_joint_pos,
                motor_joint_vel,
                base_rotation,
                base_velocities,
                height.unsqueeze(-1),
                self._contact_states,
            ),
            dim=-1,
        )

        nan_mask = torch.isnan(new_obs).any(dim=1)
        new_obs[nan_mask] = 0.0
        self._collision_buf = self._collision_buf.logical_or(nan_mask)

        self.obs_buf = new_obs.clone()
        observations = {self._olympusses.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self) -> None:
        """
        Prepares the quadroped for the next physichs step.
        NB this has to be done before each call to world.step().
        NB this method does not accept control signals as input,
        please see the apply_contol method.
        """
        #spring_actions = self._spring.forward()
        #self._olympusses.apply_action(spring_actions)
        return
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
        self._steps_since_landing_buf[self._stage_buf==2] += 1
        if self._env._world.is_playing():
            self.get_observations()
            self.get_states()
            self.is_done()  # call is done to update reset buffer before calculating metric in order to hand out termination rewards
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def apply_control(self, actions) -> None:
        """
        Apply control signals to the quadropeds.
        Args:
        actions: Tensor of shape (num_envs,4)
        """

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # lineraly interpolate between min and max
        new_targets = (0.5 * (self._motor_joint_upper_limits - self._motor_joint_lower_limits) * actions +
                                        0.5 * (self._motor_joint_lower_limits + self._motor_joint_upper_limits))
        
        #apply ema filter
        self._current_policy_targets = self._ema_alpha * new_targets + (1-self._ema_alpha) * self._current_policy_targets

        #clamp targets to joint limits and to collision free config
        self._current_action = actions.clone()  
        self._current_clamped_targets = self._clamp_joint_angels(self._current_policy_targets)
        self._olympusses.set_joint_position_targets(self._current_clamped_targets, joint_indices=self._actuated_indicies)

    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        dof_pos = self.default_articulated_joints_pos[env_ids]
        dof_vel = torch.zeros((num_resets, self._olympusses.num_dof), device=self._device)
        root_pos = self.initial_root_pos[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        
        #sample joints
        front_transversal = (torch.rand((num_resets * 4,), device=self._device)*(100-5) +5).deg2rad()
        back_transversal = (torch.rand((num_resets * 4,), device=self._device)*(100-5) +5).deg2rad()
        k_outer, k_inner, _ = self._forward_kin._calculate_knee_angles(front_transversal, back_transversal)
        lateral = (torch.rand((num_resets * 4,), device=self._device)*(-100-10) +10).deg2rad()
        #Set initial joint states
        dof_pos[:, self.front_transversal_indicies] = front_transversal.view(num_resets, 4) 
        dof_pos[:, self.back_transversal_indicies] = back_transversal.view(num_resets, 4)
        dof_pos[:, self._knee_outer_indicies] = k_outer.view(num_resets, 4)
        dof_pos[:, self._knee_inner_indicies] = k_inner.view(num_resets, 4)
        dof_pos[:, self.lateral_indicies] = lateral.view(num_resets, 4)
        
        #sample initial base state
        vel_mag = torch.rand((num_resets), device=self._device)*(4)
        theta = (torch.rand((num_resets), device=self._device)*(20)-10).deg2rad()
        phi = -torch.rand((num_resets), device=self._device)*torch.pi/4 - torch.pi/4
        vel_z = vel_mag*torch.sin(phi)
        vel_xy = vel_mag*torch.cos(phi)
        vel_x = vel_xy*torch.cos(theta)
        vel_y = vel_xy*torch.sin(theta)
        lin_vel = torch.stack((vel_x, vel_y, vel_z), dim=1)
        init_height = torch.rand(num_resets, device=self._device)*(0.5) + 1
        roll = (torch.rand(num_resets, device=self._device)*20 -10).deg2rad()
        pitch = (torch.rand(num_resets, device=self._device)*20 -10).deg2rad()
        yaw = (torch.rand(num_resets, device=self._device)*360).deg2rad()
        root_rot = quat_from_euler_xyz(roll, pitch, yaw)
        #set initial base state
        root_pos[:, 2] = init_height
        root_vel[:, :3] = lin_vel

        indices = env_ids.to(dtype=torch.int32)
        self._olympusses.set_joint_positions(dof_pos, indices)
        self._olympusses.set_joint_velocities(dof_vel, indices)
        self._olympusses.set_world_poses(root_pos, root_rot, indices)
        self._olympusses.set_velocities(root_vel, indices)
        # Bookkeeping
        self._current_clamped_targets[env_ids] = dof_pos[:,self._actuated_indicies]
        self._current_policy_targets[env_ids] = dof_pos[:,self._actuated_indicies]
        self.reset_buf[env_ids] = False
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_motor_joint_vel[env_ids] = 0.0
        
    
    def post_reset(self):
        self._forward_kin = OlympusForwardKinematics(self._device)
        self._spring = OlympusSpringJIT(k=400, olympus_view=self._olympusses, equality_dist=0.2, pulley_radius=0.02)
        self._body_inertia = self._olympusses.get_body_inertias()[0,0].reshape(3,3)
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
        self._sym_0_indicies = self.back_transversal_indicies[0:2]
        self._sym_1_indicies = self.front_transversal_indicies[0:2]
        self._sym_2_indicies = self.back_transversal_indicies[2:4]
        self._sym_3_indicies = self.front_transversal_indicies[2:4]
        self._sym_4_indicies = self.lateral_indicies[0:2]
        self._sym_5_indicies = self.lateral_indicies[2:4]
        # joimt limits
        self._motor_joint_lower_limits = torch.zeros(
            (1,self._num_actuated,), device=self._device, dtype=torch.float
        )
        self._motor_joint_upper_limits = torch.zeros(
            (1,self._num_actuated,), device=self._device, dtype=torch.float
        )
        self._motor_joint_lower_limits[0,self.front_transversal_indicies] = self.transversal_motor_limits[0]
        self._motor_joint_lower_limits[0,self.back_transversal_indicies] = self.transversal_motor_limits[0]
        self._motor_joint_lower_limits[0,self.lateral_indicies] = self.lateral_motor_limits[0]
        self._motor_joint_upper_limits[0,self.front_transversal_indicies] = self.transversal_motor_limits[1]
        self._motor_joint_upper_limits[0,self.back_transversal_indicies] = self.transversal_motor_limits[1]
        self._motor_joint_upper_limits[0,self.lateral_indicies] = self.lateral_motor_limits[1]
        # initialize buffers
        self.initial_root_pos, self.initial_root_rot = self._olympusses.get_world_poses()
        self._current_clamped_targets = self.default_actuated_joints_pos.clone()
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_motor_joint_vel = torch.zeros(
            (self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_vel = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self.last_actions = torch.zeros(
            (self._num_envs, self.num_actions), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.time_out_buf = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._is_initilized_buf =  torch.zeros(self._num_envs, dtype=torch.bool, device=self._device) #after restet wait until the agent is still on the ground befoe appllying actions
        self._init_dof_pos_buf = torch.zeros((self._num_envs, self._num_articulated_joints), dtype=torch.float, device=self._device)
        self.obs_buf = torch.zeros((self._num_envs, self._num_observations), dtype=torch.float, device=self._device)
        self._last_contact_state = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device)
        self._min_height_buf = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)
        self._curriculum_level = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._curriculum_step = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._max_height_buf = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)
        self._stage_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device) #0: stance, 1: takeoff, 2: flight, 3: landing
        self._takeoff_buf = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._land_pos_buf = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device)
        self._steps_since_landing_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._steps_since_takeoff_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._target_height = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)
        self._current_clamped_targets = torch.zeros((self._num_envs, self._num_actuated), dtype=torch.float, device=self._device)
        self._current_policy_targets = torch.zeros((self._num_envs, self._num_actuated), dtype=torch.float, device=self._device)
        # reset all envs
        indices = torch.arange(self._olympusses.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._is_test:
            self._curriculum_level +=2

    def calculate_metrics(self) -> None:
        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)
        velocity = self._olympusses.get_linear_velocities(clone=False)
        ang_velocity = self._olympusses.get_angular_velocities(clone=False)
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self._actuated_indicies)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self._actuated_indicies)

        orient_error = quat_diff_rad(base_rotation,self._zero_rotation)
        rew_orientation = exp_kernel_1d(orient_error,0.5)*10
       
        rew_vel = -velocity.norm(dim=1)*0.1
        rew_ang_vel = -ang_velocity.norm(dim=1)*0.1
        rew_bounce = -velocity[:,2].clamp(min=0)*100

        torques = ((self._current_clamped_targets - motor_joint_pos) * self.Kp - motor_joint_vel * self.Kd).clamp(min=-self.max_torque, max=self.max_torque)
        power = (torques * motor_joint_vel).abs()
        rew_power = (200 - power).mean(dim=1)/300*1
        z_height = (base_position[:,2] -0.3).abs().clamp(min=0.15)
        rew_base_height = exp_kernel_1d(z_height-0.15,0.1)*10

        rew_stepping = -(self._contact_states - self._last_contact_state).norm(dim=1)*10

        rew_survive = torch.ones(self._num_envs, device=self._device)*0
        rew_collision = -self._collision_buf.float()*50
        joint_acc = (motor_joint_vel - self.last_motor_joint_vel) / self._step_dt
        rew_joint_acc = -((joint_acc.abs()-0.01).clamp(min=0)**2).sum(dim=-1)* 0.00000001# self.rew_scales["r_joint_acc"]
        rew_contact = (self._contact_states==1).all(dim=1).float()*1
        rew_paw_height = exp_kernel_1d(self._paw_height.mean(dim=1),0.2)*10
        total_rew = rew_orientation + rew_vel + rew_power + rew_survive + rew_collision + rew_joint_acc + rew_contact + rew_paw_height  + rew_ang_vel + rew_base_height + rew_stepping+rew_bounce

        self._last_motor_joint_vel = motor_joint_vel.clone()

       
        self.rew_buf = total_rew.clone()

        self.extras["rew_orientation"] = rew_orientation.mean()
        self.extras["rew_stepping"] = rew_stepping.mean()
        self.extras["rew_vel"] = rew_vel.mean()
        self.extras["rew_power"] = rew_power.mean()
        self.extras["rew_survive"] = rew_survive.mean()
        self.extras["rew_collision"] = rew_collision.mean()
        self.extras["rew_joint_acc"] = rew_joint_acc.mean()
        self.extras["rew_contact"] = rew_contact.mean()
        self.extras["rew_paw_height"] = rew_paw_height.mean()
        self.extras["rew_ang_vel"] = rew_ang_vel.mean()
        self.extras["rew_base_height"] = rew_base_height.mean()
        self.extras["rew_bounce"] = rew_bounce.mean()





       
    
    def is_done(self) -> None:
  
        # reset agents
        
        time_out = (self.progress_buf >= self.max_episode_length - 1)
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self._actuated_indicies)
        motor_joint_pos_clamped = self._clamp_joint_angels(motor_joint_pos)
        motor_joint_violations = (torch.abs(motor_joint_pos - motor_joint_pos_clamped) > torch.pi/180).any(dim=1)
        self._collision_buf = self._collision_buf.logical_or(motor_joint_violations)
        self.reset_buf[:] = time_out.logical_or(self._collision_buf)

         

    def _clamp_joint_angels(self, joint_targets: Tensor):
        clamped_targets = joint_targets.clamp(
           self._motor_joint_lower_limits,self._motor_joint_upper_limits
        )
        #"project" to collision free state

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

    def _get_torque_limits(self,joint_velocities: Tensor) -> Tensor:
        a = -self.max_torque/(self._motor_cutoff_speed-self._motor_degration_speed)
        b = -a*self._motor_cutoff_speed
        return (a*joint_velocities.abs() + b).clamp(min=0,max=self.max_torque)
    
    def _init_config(self,i) -> Tuple[Tensor,Tensor]:
        #sample squat angle
        lower = self._curriculum_init_squat_angle_lower
        upper = self._curriculum_init_squat_angle_upper
        squat_angles = sample_squat_angle(lower, upper)
        k_outer, k_inner, init_heights = self._forward_kin.get_squat_configuration(squat_angles)
        #sample height target 
        targets = 2*torch.rand((self._num_envs), device=self._device) + 1.5
        


#########################################################
#==================== JIT FUNCTIONS ====================#
#########################################################

@torch.jit.script
def exp_kernel_3d(x: Tensor, sigma: float) -> Tensor:
    return torch.exp(-(torch.sum(x**2,dim=1) / sigma**2))
@torch.jit.script
def exp_kernel_1d(x: Tensor, sigma: float) -> Tensor:
    return torch.exp(-(x/sigma)**2)
@torch.jit.script
def laplacian_kernel_1d(x: Tensor, sigma: float) -> Tensor:
    b = sigma / torch.sqrt(2)
    return torch.exp(-torch.abs(x)/b)
    

@torch.jit.script
def estimate_jump_lenght_x(velocity: Tensor, g: float) -> Tensor:
    t = 2*velocity[:,2].clamp(min=0)/g
    x_vel = velocity[:,0].clamp(min=0)
    return x_vel*t
@torch.jit.script
def estimate_jump_lenght_y(velocity: Tensor, g: float) -> Tensor:
    t = 2*velocity[:,2].clamp(min=0)/g
    y_vel = torch.abs(velocity[:,1])
    return y_vel*t

@torch.jit.script
def estimate_jump_height(velocity: Tensor,pos: Tensor,g: float) -> Tensor:
    return (velocity[:,2].clamp(min=0)**2)/(2*g) + pos[:,2]

@torch.jit.script
def calculate_exit_angle(velocity: Tensor) -> Tensor:
    planar_vel = torch.norm(velocity[:,:2], dim=1)
    return torch.atan2(velocity[:,2], planar_vel)

@torch.jit.script
def sample_squat_angle(lower: Tensor, upper: Tensor) -> Tensor:
    return torch.rand(lower.shape[0], device=lower.device) * (upper - lower) + lower

@torch.jit.script
def sample_height_target(lower: Tensor, upper: Tensor) -> Tensor:
    return torch.rand(lower.shape[0], device=lower.device) * (upper - lower) + lower
