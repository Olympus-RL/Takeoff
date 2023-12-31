# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time

from typing import Optional,Tuple
import torch
from torch import Tensor

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class OlympusView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "OlympusView",
        track_contact_forces=True,
        prepare_contact_sensors=True,
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/Body",
            name="body_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

        self.MotorHousing_FL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/MotorHousing_FL",
            name="MotorHousing_FL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontMotor_FL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontThigh_FL",
            name="FrontMotor_FL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackMotor_FL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackThigh_FL",
            name="BackMotor_FL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontKnee_FL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontShank_FL",
            name="FrontKnee_FL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackKnee_FL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackShank_FL",
            name="BackKnee_FL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=track_contact_forces,
        )
        self.Paw_FL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/Paw_FL",
            name="Paw_FL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
            #contact_filter_prim_paths_expr= ["/World/defaultGroundPlane"],
        )
        self.MotorHousing_FR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/MotorHousing_FR",
            name="MotorHousing_FR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontMotor_FR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontThigh_FR",
            name="FrontMotor_FR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackMotor_FR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackThigh_FR",
            name="BackMotor_FR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontKnee_FR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontShank_FR",
            name="FrontKnee_FR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackKnee_FR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackShank_FR",
            name="BackKnee_FR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.Paw_FR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/Paw_FR",
            name="Paw_FR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
            #contact_filter_prim_paths_expr= ["/World/defaultGroundPlane"],
        )

        self.MotorHousing_BL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/MotorHousing_BL",
            name="MotorHousing_BL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontMotor_BL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontThigh_BL",
            name="FrontMotor_BL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackMotor_BL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackThigh_BL",
            name="BackMotor_BL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontKnee_BL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontShank_BL",
            name="FrontKnee_BL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackKnee_BL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackShank_BL",
            name="BackKnee_BL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.Paw_BL = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/Paw_BL",
            name="Paw_BL",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
            #contact_filter_prim_paths_expr= ["/World/defaultGroundPlane"],
        )

        self.MotorHousing_BR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/MotorHousing_BR",
            name="MotorHousing_BR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontMotor_BR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontThigh_BR",
            name="FrontMotor_BR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackMotor_BR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackThigh_BR",
            name="BackMotor_BR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.FrontKnee_BR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/FrontShank_BR",
            name="FrontKnee_BR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.BackKnee_BR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/BackShank_BR",
            name="BackKnee_BR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.Paw_BR = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Olympus/Paw_BR",
            name="Paw_BR",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
            #contact_filter_prim_paths_expr= ["/World/defaultGroundPlane"],
        )

        self.rigid_prims = [
            self._base,
            self.MotorHousing_FL,
            self.FrontMotor_FL,
            self.BackMotor_FL,
            self.FrontKnee_FL,
            self.BackKnee_FL,
            #self.Paw_FL,
            self.MotorHousing_FR,
            self.FrontMotor_FR,
            self.BackMotor_FR,
            self.FrontKnee_FR,
            self.BackKnee_FR,
            #self.Paw_FR,
            self.MotorHousing_BL,
            self.FrontMotor_BL,
            self.BackMotor_BL,
            self.FrontKnee_BL,
            self.BackKnee_BL,
            #self.Paw_BL,
            self.MotorHousing_BR,
            self.FrontMotor_BR,
            self.BackMotor_BR,
            self.FrontKnee_BR,
            self.BackKnee_BR,
            #self.Paw_BR,
        ]
        self._paws = [self.Paw_FL, self.Paw_FR,self.Paw_BL, self.Paw_BR]
        self._knees = [self.FrontKnee_FL, self.BackKnee_FL, self.FrontKnee_FR, self.BackKnee_FR, self.FrontKnee_BL, self.BackKnee_BL, self.FrontKnee_BR, self.BackKnee_BR]


    def get_knee_transforms(self):
        return self._knees.get_world_poses()

    def is_base_below_threshold(self, threshold, ground_heights):
        base_pos, _ = self.get_world_poses()
        base_heights = base_pos[:, 2]
        base_heights -= ground_heights

        return base_heights[:] < threshold

    def get_contact_state_collisionbuf(self) -> Tuple[Tensor, Tensor]:
        #coll_buf = torch.zeros(self._count, dtype=torch.bool, device=self._device)
        #paw_count = 0
        #for rigid_prim in self.rigid_prims:
        #    forces: Tensor = rigid_prim.get_net_contact_forces(clone=track_contact_forces=True)
        #    prim_in_collision = (forces.abs() > 1e-5).any(dim=-1)
        #    if "Paw" in rigid_prim.name:
        #        contact_state[:,paw_count] = prim_in_collision.float()
        #        paw_count += 1
        #    else:
        #        coll_buf = coll_buf.logical_or(prim_in_collision)
        #return contact_state, coll_buf
        coll_buf = self.get_collision_buf()
        contact_state = self.get_contact_state()
        #contact_state = torch.zeros(self._count,4, dtype=torch.float, device=self._device) + 1

        return contact_state, coll_buf
    
    def get_collision_buf(self) -> Tensor:
        coll_buf = torch.zeros(self._count, dtype=torch.bool, device=self._device)
        for rigid_prim in self.rigid_prims:
            forces: Tensor = rigid_prim.get_net_contact_forces(clone=True)
            prim_in_collision = (forces.abs() > 1e-5).any(dim=-1)
            coll_buf = coll_buf.logical_or(prim_in_collision)
        return coll_buf
    
    def get_contact_state(self) -> Tensor:
        contact_state = torch.cat(
            [torch.any(paw.get_net_contact_forces(clone=True).abs() > 1e-5,dim=-1).float().unsqueeze(1) for paw in self._paws],
            dim=-1
        )
        #contact_state = torch.cat(
        #    [torch.any(paw.get_contact_force_matrix(clone=False)[:,0,:].abs() > 1e-5,dim=-1).float().unsqueeze(1) for paw in self._paws],
        #    dim=-1
        #)
        return contact_state
    
    def get_paw_heights(self) -> Tensor:
        return torch.cat([paw.get_world_poses()[0][:,[2]] for paw in self._paws],dim=-1)
    def get_knee_heights(self) -> Tensor:
        return torch.cat([knee.get_world_poses()[0][:,[2]] for knee in self._knees],dim=-1)