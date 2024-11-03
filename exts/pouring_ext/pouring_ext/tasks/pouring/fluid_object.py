from omni.physx.scripts import physicsUtils, particleUtils, utils
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np
import omni.kit.commands
import torch

class FluidObjectCfg():

    # Number of particles along the hoorizontal and vertical axes (for direct spawn)
    numParticlesX: int
    numParticlesY: int
    numParticlesZ: int
    particleSpacing: float
     
    # Fluid properties
    particle_mass : float
    density : float

     

class FluidObject():

    cfg: FluidObjectCfg

    def __init__(self, cfg: FluidObjectCfg, lower_pos: Gf.Vec3f):
        self.cfg = cfg
        self.lower_pos = lower_pos # Lower position for the spawn
        
        # Scene infos (Default values)
        self.stage = get_current_stage()
        self.default_prim = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World")).GetPrim()
        self.stage.SetDefaultPrim(self.default_prim)
        self.default_prim_path = self.stage.GetDefaultPrim().GetPath()
        self.scenePath = Sdf.Path("/physicsScene")
        
    

    def spawn_fluid_direct(self):
            
            # Particle System
            self.particleSystemPath = self.default_prim_path.AppendChild("particleSystem")

            # Particle points
            self.particlesPath = Sdf.Path("/World/envs/env_0/particles")

            # solver iterations
            self._solverPositionIterations = 4
            physxAPI = PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath(self.scenePath))
            physxAPI.CreateSolverTypeAttr("TGS")

            # particle params
            restOffset = self.cfg.particleSpacing * 0.9
            fluidRestOffset = restOffset * 0.6
            particleContactOffset = restOffset + 0.001
            particle_system = particleUtils.add_physx_particle_system(
                stage=self.stage,
                particle_system_path=self.particleSystemPath,
                simulation_owner=self.scenePath,
                contact_offset=restOffset * 1.5 + 0.01,
                rest_offset=restOffset * 1.5,
                particle_contact_offset=particleContactOffset,
                solid_rest_offset=0.0,
                fluid_rest_offset=fluidRestOffset,
                solver_position_iterations=self._solverPositionIterations,
            )

            mtl_created = []
            omni.kit.commands.execute(
                "CreateAndBindMdlMaterialFromLibrary",
                mdl_name="OmniSurfacePresets.mdl",
                mtl_name="OmniSurface_DeepWater",
                mtl_created_list=mtl_created,
            )
            pbd_particle_material_path = mtl_created[0]
            omni.kit.commands.execute(
                "BindMaterial", prim_path=self.particleSystemPath, material_path=pbd_particle_material_path
            )

            # Create a pbd particle material and set it on the particle system
            particleUtils.add_pbd_particle_material(
                self.stage,
                pbd_particle_material_path,
                cohesion=10,
                viscosity=0.91,
                surface_tension=0.74,
                friction=0.1,
            )
            physicsUtils.add_physics_material_to_prim(self.stage, particle_system.GetPrim(), pbd_particle_material_path)

            particle_system.CreateMaxVelocityAttr().Set(200)

            # add particle anisotropy
            anisotropyAPI = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
            anisotropyAPI.CreateParticleAnisotropyEnabledAttr().Set(True)
            aniso_scale = 5.0
            anisotropyAPI.CreateScaleAttr().Set(aniso_scale)
            anisotropyAPI.CreateMinAttr().Set(1.0)
            anisotropyAPI.CreateMaxAttr().Set(2.0)

            # add particle smoothing
            smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())
            smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(True)
            smoothingAPI.CreateStrengthAttr().Set(0.5)

            # apply isosurface params
            isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
            isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
            isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
            isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
            isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 4)
            isosurfaceAPI.CreateGridSpacingAttr().Set(fluidRestOffset * 1.5)
            isosurfaceAPI.CreateSurfaceDistanceAttr().Set(fluidRestOffset * 1.6)
            isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
            isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(fluidRestOffset * 2)

            isosurfaceAPI.CreateNumMeshSmoothingPassesAttr().Set(1)

            primVarsApi = UsdGeom.PrimvarsAPI(particle_system)
            primVarsApi.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

            self.stage.SetInterpolationType(Usd.InterpolationTypeHeld)

            # Create Grid
            gridSpacing = self.cfg.particleSpacing + 0.001
            lower = self.lower_pos + Gf.Vec3f(-gridSpacing*self.cfg.numParticlesX/2, -gridSpacing*self.cfg.numParticlesY/2, 0) # Translate lower corner
            positions, velocities = particleUtils.create_particles_grid(
                lower, gridSpacing, self.cfg.numParticlesX, self.cfg.numParticlesY, self.cfg.numParticlesZ
            )

            uniform_range = self.cfg.particleSpacing * 0.2

            for i in range(len(positions)):
                positions[i][0] += np.random.default_rng(45).uniform(-uniform_range, uniform_range)
                positions[i][1] += np.random.default_rng(45).uniform(-uniform_range, uniform_range)
                positions[i][2] += np.random.default_rng(45).uniform(-uniform_range, uniform_range)
            widths = [self.cfg.particleSpacing] * len(positions)
            
            self.particlesPrim = particleUtils.add_physx_particleset_points(
                stage=self.stage,
                path=self.particlesPath,
                positions_list=Vt.Vec3fArray(positions),
                velocities_list=Vt.Vec3fArray(velocities),
                widths_list=widths,
                particle_system_path=self.particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=self.cfg.particle_mass,
                density=self.cfg.density,
            )

            # Hide particles
            visibility_attribute = self.particlesPrim.GetVisibilityAttr()
            visibility_attribute.Set("invisible")


    def get_particles_position(self)->tuple[Gf.Vec3f, Gf.Vec3f]:
        # Gets particles' positions and velocities and outputs them as tensors
        particles = UsdGeom.Points(self.stage.GetPrimAtPath(self.default_prim_path.AppendPath("envs/env_0/particles")))
        particles_pos = particles.GetPointsAttr().Get()
        particles_vel = particles.GetVelocitiesAttr().Get()

        return particles_pos, particles_vel

    def set_particles_position(self, particles_pos:Gf.Vec3f, particles_vel:Gf.Vec3f, env_id:int):
        # Sets the particles' position and velocities to the given arrays
        particles = UsdGeom.Points(self.stage.GetPrimAtPath(self.default_prim_path.AppendPath("envs/env_%d/particles" % env_id)))
        particles_pos = particles.GetPointsAttr().Set(particles_pos)
        particles_vel = particles.GetVelocitiesAttr().Set(particles_vel)




