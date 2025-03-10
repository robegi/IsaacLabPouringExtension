from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform

from omni.physx.scripts import physicsUtils, particleUtils, utils
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np
import omni.kit.commands
from omni.physx import acquire_physx_interface

from pouring_ext.tasks.pouring.fluid_object import FluidObjectCfg, FluidObject
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg, save_images_to_file
import omni.replicator.core as rep
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.core import PhysicsContext
import math
import quaternion
import carb.settings
import os
from omni.isaac.lab.utils import convert_dict_to_backend
import cv2
from std_msgs.msg import String
import matplotlib as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from omni.isaac.lab.sim import SimulationContext 
from copy import deepcopy
import time
import gymnasium as gym
from omni.isaac.lab.utils.math import sample_uniform

import argparse
from omegaconf import OmegaConf

from .pourit_utils.predictor import LiquidPredictor

@configclass
class FrankaPouringEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10  # 100 timesteps
    decimation = 15
    action_space = 2
    state_space = 0
    num_channels = 1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx = sim_utils.PhysxCfg(gpu_max_particle_contacts=2**22)
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=False)

    # path
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -1.4460,
                "panda_joint2": 0.2157,
                "panda_joint3": 1.2273,
                "panda_joint4": -2.4090,
                "panda_joint5": 2.8540,
                "panda_joint6": 2.2554,
                "panda_joint7": 0.7622, 
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0),
            rot=(0.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=870.0,
                velocity_limit=2.175,
                stiffness=800.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=120.0,
                velocity_limit=2.61,
                stiffness=800.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # camera
    camera_pos = (1.0, 0.1, 0.2)
    camera_rot = (0, 0, 0,  0.1)
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera_pos, rot=camera_rot, convention="world"),
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=150,
        height=150,
    )
    # observation_space = [camera.height, camera.width, num_channels] if not using PourIt
    # NOTE PourIt always crops the image to 480x480. Channels first in pytorch network. Position is of the EE relative to the target container
    observation_space = {"camera": [num_channels, camera.width, camera.height], "position": 4}


    # Joint names to actuate along the arm
    robot_arm_names = list()
    for i in range(1,8):
        robot_arm_names.append("panda_joint%d"%i)

    # Joint names of the fingers
    robot_finger_names = ["panda_finger_joint.*"]

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Spawn position for both the glass, the container and the fluid
    spawn_pos_glass = Gf.Vec3f(0.61, -0.1, 0.25)
    spawn_pos_fluid = spawn_pos_glass + Gf.Vec3f(0.0,0,0.05)
    spawn_pos_container = Gf.Vec3f(0.61, 0., 0.01)

    # Set Glass as rigid object
    glass = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Glass",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pos_glass, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{CURRENT_PATH}/usd_models/Tall_Glass_5.usd",
            semantic_tags=[("class","Glass")],
            scale=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Set target container as rigid object
    # Container data from original usd model
    container_height = 0.12
    container_radius = 0.08/2
    container_base = 0.02

    container = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Container",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pos_container, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{CURRENT_PATH}/usd_models/Tall_Glass_5.usd",
            semantic_tags=[("class","Container")],
            scale=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Add liquid configuration parameters
    # Direct spawn
    liquidCfg = FluidObjectCfg()
    liquidCfg.numParticlesX = 8
    liquidCfg.numParticlesY = 8
    liquidCfg.numParticlesZ = 44
    liquidCfg.density = 0.0
    liquidCfg.particle_mass = 0.001
    liquidCfg.particleSpacing = 0.005
    liquidCfg.viscosity = 0.91

    # Fill levels inside the source container
    particles_init_pos_list = ["particle_init_pos_low", "particle_init_pos_mid", "particle_init_pos_high"]

    # reward scales
    inside_weight = 1.0
    outside_weight = -1.0
    source_pos_weight = 0.
    source_ground_weight = -0
    source_vel_weight = -0.00
    joint_vel_weight = 0
    actions_weight = -0.1

    # Action scales
    action_scale_lin = 0.01
    action_scale_rot = 0.2



class FrankaPouringEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaPouringEnvCfg    

    def __init__(self, cfg: FrankaPouringEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints-2), device=self.device)

        self._robot_arm_idx, _ = self._robot.find_joints(self.cfg.robot_arm_names)
        self._robot_finger_idx = self._robot.find_joints(self.cfg.robot_finger_names)

        self.stage = get_current_stage()

        self.action_constraints_low = torch.tensor([-0.3, -math.pi], device=self.device)
        self.action_constraints_high = torch.tensor([0.3, 0], device=self.device)


    def _setup_scene(self):       
        
        # Force GPU dynamics to simulate liquids
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)

        # Set partial rendering
        Sim_Context = SimulationContext()
        rendermode = Sim_Context.RenderMode.FULL_RENDERING
        Sim_Context.set_render_mode(mode=rendermode)

        # Set translucency to render transparent materials
        settings = carb.settings.get_settings()
        settings.set("/rtx/translucency/enabled", True)

        # Liquid, spawns it and gets the initial positions and velocities
        self.liquid = FluidObject(cfg=self.cfg.liquidCfg, 
                             lower_pos = self.cfg.spawn_pos_fluid)
        self.liquid.spawn_fluid_direct()

        # # Initial particle position, from spawn or from saved file
        # self.liquid_init_pos, self.liquid_init_vel = torch.tensor(self.liquid.get_particles_position(0) , device = self.device)
        self.liquid_init_pos = list()
        self.liquid_init_vel = list()

        for i in range(len(self.cfg.particles_init_pos_list)):
            self.liquid_init_pos.append(torch.load(f"{self.cfg.CURRENT_PATH}/usd_models/{self.cfg.particles_init_pos_list[i]}.pt").cpu().numpy())
            self.liquid_init_pos[i] += np.ones_like(self.liquid_init_pos[i])*np.array([0, 0, 0.01])
            self.liquid_init_vel.append(np.zeros_like(self.liquid_init_pos[i]))
        
        # Reward and observations
        self.reward = np.zeros((self.num_envs))
        self.obs_reward_in = np.zeros((self.num_envs))
        self.obs_reward_out = np.zeros((self.num_envs))
        self.particle_fraction_in = np.zeros((self.num_envs,1))
        self.particle_fraction_out = np.zeros((self.num_envs,1))
        
        # Glass, position it before the robot
        self._glass = RigidObject(self.cfg.glass)
        self.scene.rigid_objects["glass"] = self._glass

        # Container
        self._container = RigidObject(self.cfg.container)
        self.scene.rigid_objects["container"] = self._container

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)     

        # Camera
        self._camera = TiledCamera(self.cfg.camera) 
        self.data_type = 'rgb'
        self.scene.sensors["camera"] = self._camera 

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # End effector controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.device)
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=self.cfg.robot_arm_names, body_names=["panda_hand"])
        self.ik_commands = torch.zeros(self.scene.num_envs, 7, device=self.device)

        # Setup for the end effector control
        self.start_ee_pos = torch.tensor([0.5, -0.1, 0.3, 0.707, 0, 0.707, 0], device=self.device) 
        self.actions_raw = torch.zeros((self.num_envs,self.cfg.action_space), device=self.device)
        self.actions_new = torch.ones((self.num_envs,7), device=self.device)*self.start_ee_pos # Starting EE position
        self.actions_total = torch.zeros((self.num_envs,self.cfg.action_space), device=self.device)
        self.deltas = torch.zeros((self.num_envs, 3), device = self.device)
        self.betas = torch.zeros((self.num_envs,3), device = self.device) 
        self.quat = torch.zeros((self.num_envs, 4), device = self.device)

        self.betas[:,0] = 1.0 # The rotation axis is fixed

         # Marker on the end effector and the desired pose
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        self._robot.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3)) # NOTE: It deactivates forces at end effector to obtain correct kinematics

        # Target on the finger actuators to hold the glass
        self.ee_target = torch.zeros((self.num_envs, 2), device = self.device)  

        # Create replicator writer
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
        self.rep_writer = rep.BasicWriter(
            output_dir=self.output_dir,
            frame_padding=0,
            colorize_instance_id_segmentation=self._camera.cfg.colorize_instance_id_segmentation,
            colorize_instance_segmentation=self._camera.cfg.colorize_instance_segmentation,
            colorize_semantic_segmentation=self._camera.cfg.colorize_semantic_segmentation,
        )
        # PourIt model
        # PourIt configuration
        self.args = argparse.Namespace
        self.args.config = f"{self.cfg.CURRENT_PATH}/pourit_utils/configs/pourit_seen_ours.yaml"
        self.args.pooling = "gmp"
        self.args.work_dir = None
        self.args.crop_size = 480
        self.args.model_path = f"{self.cfg.CURRENT_PATH}/pourit_utils/checkpoints/iter_014000.pth"
        self.predictor_cfg = OmegaConf.load(self.args.config)
        self.predictor_cfg.dataset.crop_size = self.args.crop_size
        if self.args.work_dir is not None:
            self.predictor_cfg.work_dir.dir = self.args.work_dir
        self.predictor = LiquidPredictor(self.predictor_cfg, self.args)
        self.obs = {"camera": torch.zeros((self.num_envs, self.cfg.num_channels, self.cfg.camera.width, self.cfg.camera.height), device = self.device), "position": torch.zeros((self.num_envs, 6), device = self.device)}


    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):

        # Actions are defined as deltas to apply to the current EE position. Rotations with quaternions, first extracted as axis and angle
        self.actions_raw = actions.clone()
        self.deltas = self.actions_raw[:,0]*self.cfg.action_scale_lin
        self.alphas = self.actions_raw[:,1]*self.cfg.action_scale_rot # Rotation angle
        # self.alphas = self.actions_raw.squeeze(1)*self.cfg.action_scale_rot # Rotation angle

        # # Imposed motions (UNCOMMENT TO POUR ON FIXED TRAJECTORY)
        # self.deltas = torch.zeros_like(self.deltas)
        # self.alphas = torch.zeros_like(self.alphas)

        # if (self.counter >= 100) & (self.counter < 120):
        #     self.deltas = torch.ones_like(self.deltas)*torch.tensor([0, 0.03,-0.]) /2   

        # if self.counter == 200:
        #     self.alphas = torch.ones_like(self.alphas)*(-math.pi/2)
        
        # # SAVE PARTICLES (Uncomment to save particles in order to obtain a cleaner initial position)
        # if self.counter == 200:
        #     particle_pos, vel = self.liquid.get_particles_position(0)
        #     torch.save(torch.tensor(particle_pos),f"{self.cfg.CURRENT_PATH}/usd_models/particle_init_pos_high.pt")

        self.counter += 1
        
        # Build the quaternion
        # Calculate half the angle
        half_angle = self.alphas / 2.0
        
        # Compute the quaternion components
        self.quat[:,0] = torch.cos(half_angle)
        self.quat[:,1] = self.betas[:,0] * torch.sin(half_angle)
        self.quat[:,2] = self.betas[:,1] * torch.sin(half_angle)
        self.quat[:,3] = self.betas[:,2] * torch.sin(half_angle)
        

        #  Apply action at the end effector 
        self.actions_new[:,1] += self.deltas
        self.actions_new[:,3:7] = self.multiply_quaternions(self.quat[:],self.actions_new[:,3:7])

        self.ik_commands[:] = self.actions_new
        self.diff_ik_controller.set_command(self.ik_commands)

        # Calculate joint movements to achieve the previous end effector position
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self._robot.data.root_state_w[:, 0:7]
        joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # Markers
        # self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(self.ik_commands[:, 0:3] + self.scene.env_origins, self.ik_commands[:, 3:7])

        # Joint positions to give to the robot as command
        self.joint_pos_des = torch.clamp(joint_pos_des, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])


    def _apply_action(self):
        self._robot.set_joint_position_target(self.joint_pos_des, joint_ids=self._robot_arm_idx)
        self._robot.set_joint_position_target(self.ee_target, joint_ids=self._robot_finger_idx[0])
        
    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.terminated = torch.tensor(self.obs_reward_out, device=self.device) > 0.5 # Reset if most liquid poured outside
        self.truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.terminated, self.truncated

    def _get_rewards(self) -> torch.Tensor:
        # Target and source position
        target_pos = self._container.data.root_pos_w - self.scene.env_origins
        source_pos = self._glass.data.root_pos_w - self.scene.env_origins
        source_vel = self._glass.data.root_lin_vel_w
        joint_vel = self._robot.data.joint_vel[:,:7]
        joint_acc = self._robot.data.joint_acc[:,:7] 

        # Compute reward for each environment
        for i in range (self.num_envs):
            pos, vel = self.liquid.get_particles_position(i)
            # Computes fractions
            self.particle_fraction_in[i,0], self.particle_fraction_out[i,0] =self.particle_fractions_func(
                target = target_pos[i].cpu().numpy(),
                particles = pos,
                limit_height = self.cfg.container_height,
                radius = self.cfg.container_radius,
            )
            
        particle_fraction_in = torch.tensor(self.particle_fraction_in, device = self.device)
        particle_fraction_out = torch.tensor(self.particle_fraction_out, device = self.device)

        self.reward = self.compute_reward(particles_inside = particle_fraction_in, 
                       particles_outside = particle_fraction_out,
                       source_pos = source_pos,
                       target_pos = target_pos,
                       source_vel = source_vel,
                       joint_vel = joint_vel,
                       actions=self.actions_raw,
                       limit_height=self.cfg.container_height,
                       inside_weight = self.cfg.inside_weight, 
                       outside_weight = self.cfg.outside_weight,
                       source_pos_weight = self.cfg.source_pos_weight,
                       source_ground_weight=self.cfg.source_ground_weight,
                       source_vel_weight = self.cfg.source_vel_weight,
                       joint_vel_weight = self.cfg.joint_vel_weight,
                       actions_weight=self.cfg.actions_weight)
        
        reward = torch.tensor(self.reward, device=self.device).squeeze(1)
        # print("Reward: "+str(reward[0]))
        # print("***")
        return reward.clone()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # Reset end effector controller
        self.robot_entity_cfg.resolve(self.scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        self.diff_ik_controller.reset()
        self.counter = 0 # Also resets counters
        self.index_image = 1
        self.index0 = 1
        self.actions_new = torch.ones_like(self.actions_new)*self.start_ee_pos # Starting EE position
        self.actions_total = torch.zeros((self.num_envs,self.cfg.action_space), device=self.device)

        # Reset the glass
        glass_init_pos = self._glass.data.default_root_state.clone()[env_ids]
        glass_init_pos[:,:3] = glass_init_pos[:,:3] + self.scene.env_origins[env_ids]
        self._glass.write_root_state_to_sim(glass_init_pos,env_ids=env_ids)

        # Reset the container
        container_init_pos = self._container.data.default_root_state.clone()[env_ids]
        container_init_pos[:,:3] = container_init_pos[:,:3] + self.scene.env_origins[env_ids]
        lower_bound = torch.tensor([0,-0.1,0],device=self.device)
        upper_bound = torch.tensor([0,0.1,0],device=self.device)
        container_init_pos[:,:3] += sample_uniform(lower_bound, upper_bound, container_init_pos[:,:3].shape, self.device) # Randomize
        self._container.write_root_state_to_sim(container_init_pos,env_ids=env_ids)

        
        # Reset the liquid 
        fill_index = torch.randint(0,3,(env_ids.size(0),))
        counter_fluid = 0
        for i in env_ids:
            init_pos = np.array(self.liquid_init_pos[fill_index[counter_fluid]])
            init_vel = np.zeros_like(init_pos)
            self.liquid.set_particles_position(init_pos, init_vel, i)
            counter_fluid += 1
        
        # Reset the robot and randomizes the initial position (to implement)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:,7:9] = 0.4
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        

    def _get_observations(self) -> dict:

        # Camera
        
        # Extract and save rgb output from camera
        camera_data = self._camera.data.output[self.data_type]
        # Choose whether to save the images or not
        images_are_being_saved = False

        if images_are_being_saved:
            self.save_image(camera_data/255.0, self.index_image, 0, "rgb")
        
        # Process image
        for i in range(self.num_envs):
            # Process the image using PourIt
            pourit_output = torch.tensor(self.predictor.inference(camera_data[i].cpu().numpy(), input_size=(self.obs["camera"].shape[2],self.obs["camera"].shape[3])), device = self.device)
        
            # Use mask as observation
            self.obs["camera"][i] = torch.tensor(pourit_output, device = self.device)
            # Save processed image in output folder
            if images_are_being_saved:
                self.save_image(pourit_output.permute([0,2,3,1]), self.index_image, i, "processed")

        self.index_image +=1 # Index for saving the images
        # Subtract the mean from the camera input
        mean_tensor = torch.mean(self.obs["camera"], dim=(2, 3), keepdim=True)
        self.obs["camera"] -= mean_tensor

        # Calculate the relative position of the source container
        # Get quantities from the environment
        source_pos = self._glass.data.root_pos_w - self.scene.env_origins
        source_rot = self._glass.data.body_quat_w
        source_vel = self._glass.data.root_lin_vel_w
        source_rot_vel = self._glass.data.root_ang_vel_w
        target_pos = self._container.data.root_pos_w - self.scene.env_origins
        
        # Scaled joint quantities
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos[:,:7] - self.robot_dof_lower_limits[:7])
            / (self.robot_dof_upper_limits[:7] - self.robot_dof_lower_limits[:7])
            - 1.0
        )
        joint_vel = self._robot.data.joint_vel[:,:7] * 0.1

        # Relative position
        relative_pos = source_pos - target_pos

        # Rotation conversion to euler
        source_rot = self.quaternion_to_euler(source_rot)[:,:,0]/torch.pi

        # Compute reward for each environment to use as observation
        for i in range (self.num_envs):
            pos, vel = self.liquid.get_particles_position(i)

            self.obs_reward_in[i], self.obs_reward_out[i] = self.particle_fractions_func(target=target_pos[i].cpu().numpy(),
                                particles=pos,
                                limit_height=self.cfg.container_height,
                                radius=self.cfg.container_radius,)
            
        obs_reward_in = torch.tensor(self.obs_reward_in, device = self.device).unsqueeze(1)
        obs_reward_out = torch.tensor(self.obs_reward_out, device = self.device).unsqueeze(1)
        
        # Concatenate observations
        self.obs["position"] = torch.cat((relative_pos[:,1].unsqueeze(1), source_rot, self.actions_raw), dim=-1).type(torch.float32)
        # self.obs["position"] = torch.cat((source_rot, self.actions_raw, obs_reward_in, obs_reward_out), dim=-1).type(torch.float32)

        # print("Source rotation: "+str(source_rot))
        # print("Observed reward in: "+str(obs_reward_in))
        # print("Observed reward out: "+str(obs_reward_out))
        # print("Relative pos: "+str(relative_pos[:,1].unsqueeze(1)))
        # print("Previous actions: "+str(self.actions_raw))
        # print("***")

        observations = {"policy": {"camera": self.obs["camera"].clone(),"position": self.obs["position"].clone()}}

        return observations

    # auxiliary methods

    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions.
        """

        w1 = q1[:, 0]
        x1 = q1[:, 1]
        y1 = q1[:, 2]
        z1 = q1[:, 3]
        w2 = q2[:, 0]
        x2 = q2[:, 1]
        y2 = q2[:, 2]
        z2 = q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = torch.unsqueeze(w,1)
        x = torch.unsqueeze(x,1)
        y = torch.unsqueeze(y,1)
        z = torch.unsqueeze(z,1)

        return torch.cat((w, x, y, z),dim=-1)

    def compute_reward(self, 
                       particles_inside: torch.tensor, 
                       particles_outside: torch.tensor,
                       source_pos: torch.tensor,
                       target_pos: torch.tensor,
                       source_vel: torch.tensor,
                       joint_vel: torch.tensor,
                       actions: torch.tensor,
                       limit_height: float,
                       inside_weight: float, 
                       outside_weight: float,
                       source_pos_weight: float,
                       source_ground_weight: float,
                       source_vel_weight: float,
                       joint_vel_weight: float,
                       actions_weight: float):
            """
            Computes the reward by considering the fraction of particles inside the target container and outside of it
            """

            # The weighted reward output is the fraction of insideoutside particles w.r.t. the total number of particles
            reward_in = inside_weight*particles_inside
            reward_out = outside_weight*particles_outside
            # reward_in = inside_weight*torch.where(particles_inside<0.5,particles_inside,0.5)
            # reward_out = outside_weight*(particles_outside+torch.where(particles_inside>0.5,particles_inside-0.5,0))

            # Penalty for source distant from target 
            dist = torch.norm(source_pos-target_pos, dim=1)
            relative_pos = source_pos-target_pos
            reward_dist = torch.zeros((self.num_envs, 1))
            reward_dist += torch.where((relative_pos[:,1]>0) | (relative_pos[:,1]<-0.25), 1., 0.).unsqueeze(1)
            reward_dist = source_pos_weight*reward_dist

            # Penalty for source glass on the ground or too low
            reward_ground = torch.zeros((self.num_envs, 1))
            reward_ground += torch.where(source_pos[:,2]<limit_height, 1., 0.).unsqueeze(1)
            reward_ground = source_ground_weight*reward_ground

            # Penalty for fast movements of the source container
            vel = torch.norm(source_vel, dim=1)
            reward_vel = torch.zeros((self.num_envs, 1))
            reward_vel += torch.where(vel>2., 1., 0.).unsqueeze(1)
            reward_vel = source_vel_weight*reward_vel

            # Penalty for joint velocities
            reward_joint_vel = torch.sum(joint_vel**2, dim=-1)
            reward_joint_vel = joint_vel_weight*reward_joint_vel.unsqueeze(1)

            # Penalty for action magnitude
            reward_actions = torch.sum(actions**2, dim=-1)
            reward_actions= actions_weight*reward_actions.unsqueeze(1)


            # print("Reward distance: "+str(reward_dist))
            # print("Reward joint vel: "+str(reward_joint_vel))
            # print("Reward actions: "+str(reward_actions))
            # print("Reward in: "+str(reward_in))
            # print("Reward out: "+str(reward_out))
            # print("***")

            reward_tot = reward_in + reward_out + reward_dist + reward_ground + reward_vel + reward_joint_vel +reward_actions

            return reward_tot
    
    def particle_fractions_func(self, 
                       target: np.array,
                       particles: np.array, 
                       limit_height: float,
                       radius: float,) -> tuple[np.array, np.array]:
            """
            Computes fraction of particles inside the target container and outside of it
            Used for both the reward and the observations
            """

            # Only considers particles below a certain limit height, which is ideally the target container's height
            index = np.where(particles[:,2]<=limit_height)

            x = particles[index,0]
            y = particles[index,1]
            z = particles[index,2]

            x_0 = target[0]
            y_0 = target[1]

            # Calculates particles inside if they are inside the round container's radius 
            num_particles = len(particles)
            index_inside = np.where(((x-x_0)**2+(y-y_0)**2 < radius**2))
            particles_inside = np.size(index_inside, 1)/num_particles

            # Calculates particles outside if they are outside the round container's radius
            index_outside = np.where((x-x_0)**2+(y-y_0)**2 >= radius**2)
            particles_outside = np.size(index_outside, 1)/num_particles

            return particles_inside, particles_outside

    def quaternion_to_euler(self, q):
        """
        Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
        
        Parameters:
        q (torch.Tensor): A tensor of shape (..., 4) where each quaternion is represented by 
                        (w, x, y, z). The last dimension should have 4 elements representing 
                        the quaternion.

        Returns:
        torch.Tensor: A tensor of shape (..., 3) representing the Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = torch.split(q, 1, dim=-1)  # Split quaternion into w, x, y, z components
        
        # Compute roll (x-axis rotation)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        
        # Compute pitch (y-axis rotation)
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        
        # Compute yaw (z-axis rotation)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        # Stack Euler angles (roll, pitch, yaw) into a tensor
        euler_angles = torch.cat((roll, pitch, yaw), dim=-1)
        
        return euler_angles

    def save_image(self, file, index_image, index_env, name):
        # Save images from camera 
        if not torch.is_tensor(file):
            file = torch.tensor(file, device=self.device)
        # Adjust dimensions
        if len(file.shape)<4:
            file = torch.unsqueeze(file, 0)
        # Expand number of channels
        if file.shape[3]==1:
            #print(file.unique())
            file_new = torch.zeros((file.shape[0],file.shape[1],file.shape[2],3), device=self.device)
            file_new[:] = file 
            file = file_new
            #print(file.unique())
        save_images_to_file(file, f"{self.cfg.CURRENT_PATH}/output/camera/{name}_{index_env}_{index_image}.png")









    
