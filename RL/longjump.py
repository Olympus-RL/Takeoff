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


class LongJumpTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._is_test = self._cfg["test"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["r_orient"] = self._task_cfg["env"]["learn"]["rOrientRewardScale"]
        self.rew_scales["r_base_acc"] = self._task_cfg["env"]["learn"]["rBaseAccRewardScale"]
        self.rew_scales["r_action_clip"] = self._task_cfg["env"]["learn"]["rActionClipRewardScale"]
        self.rew_scales["r_torque_clip"] = self._task_cfg["env"]["learn"]["rTorqueClipRewardScale"]
        self.rew_scales["r_jump"] = self._task_cfg["env"]["learn"]["rJumpRewardScale"]
        self.rew_scales["r_max_height"] = self._task_cfg["env"]["learn"]["rMaxHeightRewardScale"]
        self.rew_scales["r_pos_tracking"] = self._task_cfg["env"]["learn"]["rPosTrackingRewardScale"]
        self.rew_scales["r_accend"] = self._task_cfg["env"]["learn"]["rAccendRewardScale"]
        self.rew_scales["r_squat"] = self._task_cfg["env"]["learn"]["rSquatRewardScale"]
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
        self._springFrequencyInv = 4 #self._task_cfg["env"]["springFrequencyInv"]
        self._step_dt = self.dt * self._controlFrequencyInv
        self._memory_lenght = 1
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self._max_time_after_landing = 0.5 #seconds
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

        self._nominal_height = torch.tensor(pos[-1], device=self._device)
        self._basline_height = torch.tensor(3.0, device=self._device)
        self._target_rotation = torch.tensor([1,0,0,0], device=self._device).expand(self._num_envs, -1)
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
        
        
        is_airborne = torch.all(self._contact_states == 0, dim=1)

        next_stage = self._stage_buf.clone()
        # stance -> takeoff
        next_stage[(self._stage_buf==0).logical_and(is_airborne).logical_and(base_velocities[:, 2] > 0.5)] = 1
        # takeoff -> landed
        next_stage[(self._stage_buf==1).logical_and(~is_airborne)] = 2
        # wait for the robot to be still on the ground before starting
        self._is_initilized_buf[torch.all(self._contact_states == 1, dim=1)] = True
        # keep stance if not initialized
        next_stage[~self._is_initilized_buf] = 0
       

        self._takeoff_buf = (next_stage == 1).logical_and(self._stage_buf == 0)
        self._landed_buf = (next_stage == 2).logical_and(self._stage_buf == 1)
        self._land_pos_buf[self._landed_buf] = base_position[self._landed_buf]
        self._min_height_buf = torch.min(self._min_height_buf, height)
        self._max_height_buf = torch.max(self._max_height_buf, height)
        self._est_height_buf = estimate_jump_height(base_velocities, base_position,3.72)
        self._steps_since_takeoff_buf[self._stage_buf==1] += 1

        self._stage_buf = next_stage
        
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
        #keep old targets if not initialized or in the air
        should_update_targets = self._stage_buf==0
        self._current_clamped_targets[should_update_targets] = self._clamp_joint_angels(self._current_policy_targets)[should_update_targets]
        
        #self._current_clamped_targets[(~self._is_initilized_buf).logical_and(~(self._stage_buf==1)),:] = self._init_dof_pos_buf[~self._is_initilized_buf,:][:,self._actuated_indicies]
        

        #apply motor limits
        #self._current_torque_limits = self._get_torque_limits(self._olympusses.get_joint_velocities(clone=False, joint_indices=self._actuated_indicies))
        #self._olympusses.set_max_efforts(self._current_torque_limits, joint_indices=self._actuated_indicies)
        # Set targets
        self._olympusses.set_joint_position_targets(self._current_clamped_targets, joint_indices=self._actuated_indicies)

    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        dof_pos = self.default_articulated_joints_pos[env_ids]
        dof_vel = torch.zeros((num_resets, self._olympusses.num_dof), device=self._device)
        root_pos = self.initial_root_pos[env_ids]
        root_rot = torch.tensor([1,0,0,0],device=self._device).expand(num_resets, -1)
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        
        #sample height target 
        targets = 2*torch.rand((num_resets), device=self._device) + 1.5 #sample between 2 and 4
        self._target_height[env_ids] = targets
        
        #smaple squat angle
        lower = self._curriculum_init_squat_angle_lower[self._curriculum_level[env_ids]]
        upper = self._curriculum_init_squat_angle_upper[self._curriculum_level[env_ids]]
        squat_angles = sample_squat_angle(lower, upper)
        k_outer, k_inner, init_heights = self._forward_kin.get_squat_configuration(squat_angles)
        #sample feet height
        lower = self._curriculum_init_base_height_lower[self._curriculum_level[env_ids]]
        upper = self._curriculum_init_base_heigt_upper[self._curriculum_level[env_ids]]
        paw_heights = (upper-lower)*torch.rand((num_resets), device=self._device) + lower



        #Set initial joint states
        dof_pos[:, self._transversal_indicies] = squat_angles.unsqueeze(-1)
        dof_pos[:, self._knee_outer_indicies] = k_outer.unsqueeze(-1)
        dof_pos[:, self._knee_inner_indicies] = k_inner.unsqueeze(-1)
        root_pos[:, 2] = (init_heights + paw_heights)
        ##
        # Apply resets
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
        self._is_initilized_buf[env_ids] = False
        self._init_dof_pos_buf[env_ids] = dof_pos
        self._min_height_buf[env_ids] = root_pos[:,2]
        self._max_height_buf[env_ids] = root_pos[:,2]
        self._stage_buf[env_ids] = 0
        self._steps_since_landing_buf[env_ids] = 0
        self._steps_since_takeoff_buf[env_ids] = -1
    
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
       
        ### Task Rewards ###
        #rew_jump = exp_kernel_1d(self._est_height_buf-self._target_height,2)*400 #self.rew_scales["r_jump"]
        rew_jump = laplacian_kernel_1d((self._est_height_buf-self._target_height)/self._target_height,1)*400 #self.rew_scales["r_jump"]
        rew_jump[self._steps_since_takeoff_buf < self._max_steps_after_take_off] = 0
        rew_jump[self._collision_buf] = 0 
        rew_jump[self._est_height_buf < 0.6] = 0


      
        orient_error = quat_diff_rad(base_rotation, self._target_rotation)
        rew_orient = exp_kernel_1d(orient_error, torch.pi/2) * 10 #self.rew_scales["r_orient"]
        rew_orient[self._stage_buf != 1] = 0  # only give orient reward when flying
        rew_land_stand = exp_kernel_3d(base_position-self._land_pos_buf, 0.1) * 0.1 #self.rew_scales["r_land_stand"]
        rew_land_stand[self._stage_buf != 2] = 0  # only give land stand reward when landing
        rew_accend = (1 -torch.abs((self._target_height - self._est_height_buf)/self._target_height))*800
        
        #velocity[:,2].clamp(min=torch.zeros(self._num_envs,device=self._device,dtype=torch.float))*self._step_dt* 10000
        #rew_accend[self._stage_buf != 1] = 0  # only give accend reward when flying
        rew_accend[self._steps_since_takeoff_buf < self._max_steps_after_take_off] = 0
        rew_accend[self._collision_buf] = 0
        rew_accend[self._est_height_buf < 0.6] = 0

        exit_angle = calculate_exit_angle(velocity)
        rew_exit_angle = (3*torch.pi/180-(torch.pi/2 - exit_angle).abs())*500 #self.rew_scales["r_exit_angle"]
        rew_exit_angle[~self._takeoff_buf] = 0  # only give exit angle reward when flying

        rew_inside_threshold = ((self._est_height_buf - self._target_height).abs() < 0.02).float() * 10000
        rew_inside_threshold[self._steps_since_takeoff_buf < self._max_steps_after_take_off] = 0
        rew_inside_threshold[ang_velocity.norm(dim=1) > 0.5] = 0
        rew_inside_threshold[(exit_angle-torch.pi/2).rad2deg() > 1] = 0


        
        ### Regualization Rewards ###
        rew_lateral_pos = -torch.sum(motor_joint_pos[:,self.lateral_indicies]**2,dim=-1)*3#self.rew_scales["r_lateral_pos"]
        rew_lateral_pos[self._stage_buf==1] = 0
       
        joint_acc = (motor_joint_vel - self.last_motor_joint_vel) / self._step_dt
        rew_joint_acc = -((joint_acc.abs()-0.01).clamp(min=0)**2).sum(dim=-1)* 0.00000001# self.rew_scales["r_joint_acc"]
        rew_collision = -10*self._collision_buf.float()#this should come from config 
    
 
#
        symmetri_metric = motor_joint_pos[:,self._transversal_indicies].var(dim=1)
        rew_symmetry = exp_kernel_1d(symmetri_metric,0.2)*20
        rew_spin = (10-torch.norm(ang_velocity, dim=1))*80
        rew_spin[~self._takeoff_buf] = 0
        rew_spin[self._est_height_buf < 1] = 0

        rew_contact = (self._contact_states == 1).all(dim=1).float()
        rew_contact[~(self._stage_buf==0)] = 0

        rew_paw_height = exp_kernel_1d(torch.sum((self._paw_height.clamp(min=0.1)-0.1)**2,dim=1),0.1)*10
        rew_paw_height[self._stage_buf==1] = 0


        ### last regualization rewards ###
        torque = (self.Kp*(self._current_clamped_targets - motor_joint_pos) - self.Kd * motor_joint_vel).clamp(min=-self.max_torque, max=self.max_torque)
        power = (torque * motor_joint_vel)
        rew_power = (1-power.abs()/300).mean(dim=1)*2
        #rew_joint_vel = -((motor_joint_vel.abs()-self._motor_cutoff_speed*0.5).clamp(min=0)**2).mean(dim=1)*0.001
        rew_joint_vel = (1-motor_joint_vel.abs()/(0.5*self._motor_cutoff_speed)).mean(dim=1)*2
        rew_action_clip = -(torch.sum((self._current_policy_targets - self._current_clamped_targets)**2, dim=1))*0.1  #self.rew_scales["r_action_clip"]
        rew_action_clip[~self._is_initilized_buf] = 0
        rew_action_clip[self._stage_buf==1] = 0

        rew_last_reg = rew_power + rew_joint_vel + rew_action_clip + 100*rew_joint_acc
        rew_last_reg[~(self._curriculum_level==self._n_curriculum_levels-1)] = 0

        total_reward = (3*(rew_jump + rew_accend) + 3*rew_spin + rew_inside_threshold + rew_joint_acc+ rew_power+ rew_contact + rew_paw_height + rew_lateral_pos + rew_exit_angle + rew_symmetry) * self.rew_scales["total"]

        
       
        # Save last values
        self.last_actions = self.actions.clone()
        self.last_motor_joint_vel = motor_joint_vel.clone()
        self.last_vel = velocity.clone()
        self._last_contact_state = self._contact_states.clone()
        # Place total reward in buffer
        self.rew_buf = total_reward.detach().clone()

        # update extras
        terminate_mask = self._steps_since_takeoff_buf >= self._max_steps_after_take_off
        self.extras["detailed_rewards/paw_height"] = rew_paw_height[self._stage_buf==0].detach().mean()
        self.extras["detailed_rewards/inside_threshold"] = rew_inside_threshold.detach().mean()

        self.extras["detailed_rewards/last_reg"] = rew_last_reg.detach().mean() 
        self.extras["detailed_rewards/contact"] = rew_contact[self._stage_buf==0].detach().mean()
        self.extras["detailed_rewards/collision"] = rew_collision.detach().mean()
        self.extras["detailed_rewards/orient"] = rew_orient.detach().mean()
        self.extras["detailed_rewards/jump"]= rew_jump[terminate_mask].detach().mean()
        self.extras["detailed_rewards/symmetry"] = rew_symmetry.detach().mean()
        self.extras["detailed_rewards/spin"] = rew_spin[self._takeoff_buf].detach().mean()
        self.extras["detailed_rewards/lateral_pos"] = rew_lateral_pos[self._stage_buf!=1].detach().mean()
        self.extras["detailed_rewards/accend"] = rew_accend[terminate_mask].detach().mean()
        self.extras["detailed_rewards/power"] = rew_power.detach().mean()
        self.extras["detailed_rewards/action_clip"] = rew_action_clip.detach().mean()
        self.extras["detailed_rewards/joint_vel"] = rew_joint_vel.detach().mean()
        self.extras["detailed_rewards/joint_acc"] = rew_joint_acc.detach().mean()
        #self.extras["detailed_rewards/stepping"] = rew_stepping.detach().mean()
        self.extras["metrics/est_height_0"] = (self._est_height_buf[terminate_mask.logical_and(self._curriculum_level==0)]).mean()
        self.extras["metrics/est_height_1"] = (self._est_height_buf[terminate_mask.logical_and(self._curriculum_level==1)]).mean()
        self.extras["metrics/est_height_2"] = (self._est_height_buf[terminate_mask.logical_and(self._curriculum_level==2)]).mean()
        self.extras["metrics/est_height_3"] = (self._est_height_buf[terminate_mask.logical_and(self._curriculum_level==3)]).mean()
        self.extras["metrics/min_height"] = self._min_height_buf[terminate_mask].mean()
        self.extras["metrics/max_height"] = self._max_height_buf[terminate_mask].mean()
        self.extras["metrics/height_deviation_0"] = (self._est_height_buf-self._target_height)[terminate_mask.logical_and(self._curriculum_level==0)].abs().mean()
        self.extras["metrics/height_deviation_1"] = (self._est_height_buf-self._target_height)[terminate_mask.logical_and(self._curriculum_level==1)].abs().mean()
        self.extras["metrics/height_deviation_2"] = (self._est_height_buf-self._target_height)[terminate_mask.logical_and(self._curriculum_level==2)].abs().mean()
        self.extras["metrics/num_takeoffs"] = self._takeoff_buf.sum()
        self.extras["curriculum/level_0_fraq"] = (self._curriculum_level==0).float().mean()
        self.extras["curriculum/level_1_fraq"] = (self._curriculum_level==1).float().mean()
        self.extras["curriculum/level_2_fraq"] = (self._curriculum_level==2).float().mean()
        self.extras["curriculum/level_3_fraq"] = (self._curriculum_level==3).float().mean()
           
        
       
        self.extras["metrics/exit_angle"] = exit_angle[self._takeoff_buf].mean()
   
    
    def is_done(self) -> None:
  
        # reset agents
        if not self._is_test:
            time_out = torch.logical_or(self.progress_buf >= self.max_episode_length - 1,self._steps_since_takeoff_buf >=self._max_steps_after_take_off)
        else:
            time_out = (self.progress_buf >= self.max_episode_length - 1)
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self._actuated_indicies)
        motor_joint_pos_clamped = self._clamp_joint_angels(motor_joint_pos)
        motor_joint_violations = (torch.abs(motor_joint_pos - motor_joint_pos_clamped) > torch.pi/180).any(dim=1)
        self._collision_buf = self._collision_buf.logical_or(motor_joint_violations)
        self.reset_buf[:] = time_out.logical_or(self._collision_buf)

        #take_off = (self._steps_since_takeoff_buf ==self._max_steps_after_take_off)
        #if take_off.any():
        #    dev = (self._est_height_buf - self._target_height).abs()[take_off]
        #    print("deviations:")
        #    print("mean:", dev.mean().item())
        #    print("std:", dev.std().item())
        #    print("max:", dev.max().item())
        #    print("min:", dev.min().item())
      
        if True: #not self._is_test:
            #progress at level 0
            made_progress = ((self._est_height_buf - self._target_height).abs() <= self._curriculum_tresh).logical_and(~self._collision_buf).logical_and(self.reset_buf)
            failed = (~made_progress).logical_and(self.reset_buf)
            self._curriculum_step[made_progress] += 1
            self._curriculum_step[failed] = 0
            level_up = (self._curriculum_step >= self._steps_per_curriculum_level).logical_and(made_progress).logical_and(self._curriculum_level!=self._n_curriculum_levels-1)
            self._curriculum_level[level_up] += 1
            self._curriculum_step[level_up] = 0



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
def estimate_jump_lenght_x(velocity: Tensor, g: float,dh: Tensor) -> Tensor:
    vel_z = velocity[:,2]
    under_root = vel_z**2 + 2*g*vel_z*dh
    t = (vel_z + torch.sqrt(under_root.clamp(min=0)))/g
    x_vel = velocity[:,0].clamp(min=0)
    dx = x_vel*t
    dx[under_root < 0] = 0
    return dx

@torch.jit.script
def estimate_jump_lenght_y(velocity: Tensor, g: float, dh: Tensor) -> Tensor:
    vel_z = velocity[:,2]
    under_root = vel_z**2 + 2*g*vel_z*dh
    t = (vel_z + torch.sqrt(under_root.clamp(min=0)))/g
    y_vel = velocity[:,1].clamp(min=0)
    dy = y_vel*t
    dy[under_root < 0] = 0
    return dy

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
