from typing import List, Tuple
import os

   
import torch
from torch import Tensor

from omni.isaac.core.utils.torch.rotations import get_euler_xyz


class JumpLogger:
    def __init__(self, device: str,num_envs: int,log_dt: float,log_dir: str = "logs"):
        self._device = device
        self._num_envs = num_envs
        self._log_dt = log_dt

        self._time =-log_dt
        self._time_data = []
        self._pos_data = []
        self._linvel_data = []
        self._rot_data = []
        self._angvel_data = []
        self._roll_rate_data = []
        self._pitch_rate_data = []
        self._yaw_rate_data = []
        self._take_off_time = -torch.ones(num_envs, device=device)
      
        
    def log_point(self, pos: Tensor, rot: Tensor,linvel: Tensor, angvel: Tensor,take_off: Tensor) -> None:
        """
        Logs the given data.
        Args:
            time: The time.
            pos: The position.
            rot: The rotation.
            linvel: The linear velocity.
            angvel: The angular velocity.
            take_off: The take off signal.
        """
        self._time += self._log_dt
        self._time_data.append(self._time)
        self._pos_data.append(pos)
        self._linvel_data.append(linvel)
        self._rot_data.append(rot)
        self._angvel_data.append(angvel)
        roll_rate, pitch_rate, yaw_rate = _ang_vel_to_euler_rates(rot, angvel)
        self._roll_rate_data.append(roll_rate)
        self._pitch_rate_data.append(pitch_rate)
        self._yaw_rate_data.append(yaw_rate)
        self._take_off_time[take_off] = self._time

    
    def plot_data(self) -> None:
        """
        Plots the logged data.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        #avrage over all envs
        pos_mu = torch.stack(self._pos_data).mean(dim=1).cpu().numpy()
        linvel_mu = torch.stack(self._linvel_data).mean(dim=1).cpu().numpy()
        rot_mu = torch.stack(self._rot_data).mean(dim=1).cpu().numpy()
        angvel_mu = torch.stack(self._angvel_data).mean(dim=1).cpu().numpy()
        roll_rate_mu = torch.stack(self._roll_rate_data).mean(dim=1).cpu().numpy()
        pitch_rate_mu = torch.stack(self._pitch_rate_data).mean(dim=1).cpu().numpy()
        yaw_rate_mu = torch.stack(self._yaw_rate_data).mean(dim=1).cpu().numpy()
        
        #get variance 
        pos_var = torch.stack(self._pos_data).var(dim=1).cpu().numpy()
        linvel_var = torch.stack(self._linvel_data).var(dim=1).cpu().numpy()
        rot_var = torch.stack(self._rot_data).var(dim=1).cpu().numpy()
        angvel_var = torch.stack(self._angvel_data).var(dim=1).cpu().numpy()
        roll_rate_var = torch.stack(self._roll_rate_data).var(dim=1).cpu().numpy()
        pitch_rate_var = torch.stack(self._pitch_rate_data).var(dim=1).cpu().numpy()
        yaw_rate_var = torch.stack(self._yaw_rate_data).var(dim=1).cpu().numpy()

        #plot xz trajectory
        x = torch.stack(self._pos_data)[:,0,0].cpu().numpy()
        z = torch.stack(self._pos_data)[:,0,2].cpu().numpy()
        plt.plot(x,z)
        plt.xlabel("Translation X [m]")
        plt.ylabel("Jump Height - Translation Z [m]")
        plt.savefig(os.path.join("logs","jump_trajectory.pdf"),format="pdf")
        
        #plot euler rates
        r,p,y = torch.stack(self._roll_rate_data).rad2deg().cpu().numpy()[:,0],torch.stack(self._pitch_rate_data).rad2deg().cpu().numpy()[:,0],torch.stack(self._yaw_rate_data).rad2deg().cpu().numpy()[:,0]
        plt.plot(self._time_data,r,label="roll rate")
        plt.plot(self._time_data,p,label="pitch rate")
        plt.plot(self._time_data,y,label="yaw rate")
        #plot vertical line at take off
        takeoff_time = self._take_off_time.cpu().numpy()[0]
        plt.axvline(x=takeoff_time,color="black",linestyle="--")
        plt.legend()
        plt.xlabel("Time [s]")
        # y labe is degs per second
        plt.ylabel("Euler Rates [deg/s]")

        plt.savefig(os.path.join("logs","jump_euler_rates.pdf"),format="pdf")
    



        




@torch.jit.script
def _ang_vel_to_euler_rates(q: torch.Tensor, ang_vel: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Converts angular velocity to euler rates.
    Args:
        q: The roation quatoernion.
        ang_vel: The angular velocity.
    Returns:
        x_dot: The euler rate around the x axis.
        y_dot: The euler rate around the y axis.
        z_dot: The euler rate around the z axis.
    """
    x,y,z = get_euler_xyz(q)

    E_inv = torch.zeros((q.shape[0],3, 3), device=q.device)
    E_inv[:, 0, 0] = 1
    E_inv[:, 0, 1] = torch.sin(x) * torch.tan(y)
    E_inv[:, 0, 2] = -torch.cos(x) * torch.tan(y)
    E_inv[:, 1, 1] = torch.cos(x)
    E_inv[:, 1, 2] = torch.sin(x)
    E_inv[:, 2, 1] = -torch.sin(x) / torch.cos(y)
    E_inv[:, 2, 2] = torch.cos(x) / torch.cos(y)

    rates = torch.bmm(E_inv, ang_vel.unsqueeze(-1)).squeeze(-1)
    return rates[:, 0], rates[:, 1], rates[:, 2]
