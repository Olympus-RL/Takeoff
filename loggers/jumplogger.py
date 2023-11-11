from typing import List, Tuple
import os

   
import torch
from torch import Tensor


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
        self._roll_data = []
        self._pitch_data = []
        self._yaw_data = []
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
        roll,pitch,yaw = _quat_to_euler_zyx(rot)
        self._roll_data.append(roll)
        self._pitch_data.append(pitch)
        self._yaw_data.append(yaw)
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
        fig1, ax1 = plt.subplots()
        x = torch.stack(self._pos_data)[:,0,0].cpu().numpy()
        z = torch.stack(self._pos_data)[:,0,2].cpu().numpy()
        ax1.plot(x,z)
        ax1.set_xlabel("Translation X [m]")
        ax1.set_ylabel("Jump Height - Translation Z [m]")
        fig1.savefig(os.path.join("logs","jump_trajectory.pdf"),format="pdf")
        
        #plot euler rates
        fig2, ax2 = plt.subplots()
        r,p,y = torch.stack(self._roll_rate_data).rad2deg().cpu().numpy()[:,0],torch.stack(self._pitch_rate_data).rad2deg().cpu().numpy()[:,0],torch.stack(self._yaw_rate_data).rad2deg().cpu().numpy()[:,0]
        ax2.plot(self._time_data,r,label="roll rate")
        ax2.plot(self._time_data,p,label="pitch rate")
        ax2.plot(self._time_data,y,label="yaw rate")
        #plot vertical line at take off
        takeoff_time = self._take_off_time.cpu().numpy()[0]
        ax2.axvline(x=takeoff_time,color="black",linestyle="--")
        ax2.legend()
        ax2.set_xlabel("Time [s]")
        # y labe is degs per second
        ax2.set_ylabel("Euler Rates [deg/s]")
        fig2.savefig(os.path.join("logs","jump_euler_rates.pdf"),format="pdf")

        #plot euler angels
        fig3, ax3 = plt.subplots()
        r = torch.stack(self._roll_data).rad2deg().cpu().numpy()[:,0]
        p = torch.stack(self._pitch_data).rad2deg().cpu().numpy()[:,0]
        y = torch.stack(self._yaw_data).rad2deg().cpu().numpy()[:,0]
        #ax3.plot(self._time_data,r,label="roll")
        #ax3.plot(self._time_data,p,label="pitch")
        ax3.plot(self._time_data,y,label="yaw")
        #plot vertical line at take off
        takeoff_time = self._take_off_time.cpu().numpy()[0]
        ax3.axvline(x=takeoff_time,color="black",linestyle="--")
        ax3.legend()
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Euler Angles [deg]")
        fig3.savefig(os.path.join("logs","jump_euler_angles.pdf"),format="pdf")

        #plot angular velocity
        fig4, ax4 = plt.subplots()
        x = torch.stack(self._angvel_data).rad2deg().cpu().numpy()[:,0,0]
        y = torch.stack(self._angvel_data).rad2deg().cpu().numpy()[:,0,1]
        z = torch.stack(self._angvel_data).rad2deg().cpu().numpy()[:,0,2]
        ax4.plot(self._time_data,x,label="x")
        ax4.plot(self._time_data,y,label="y")
        ax4.plot(self._time_data,z,label="z")
        #plot vertical line at take off
        takeoff_time = self._take_off_time.cpu().numpy()[0]
        ax4.axvline(x=takeoff_time,color="black",linestyle="--")
        ax4.legend()
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Angular Velocity [deg/s]")
        fig4.savefig(os.path.join("logs","jump_angular_velocity.pdf"),format="pdf")

        print("print plotted data to logs folder")
    



        




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
    x,y,z = _quat_to_euler_zyx(q)


    E_inv = torch.zeros((q.shape[0],3, 3), device=q.device)
    E_inv[:, 0, 0] = torch.cos(z)*torch.tan(y)
    E_inv[:, 0, 1] = torch.tan(y)*torch.sin(z)
    E_inv[:, 0, 2] = 1
    E_inv[:, 1, 0] = -torch.sin(z)
    E_inv[:, 1, 1] = torch.cos(z)
    E_inv[:, 2, 0] = torch.cos(z)/torch.cos(y)
    E_inv[:, 2, 1] = torch.sin(z)/torch.cos(y)

    rates = torch.bmm(E_inv, ang_vel.unsqueeze(-1)).squeeze(-1)
    return rates[:, 2], rates[:, 1], rates[:, 0]

def _quat_to_euler_zyx(q: Tensor) -> Tuple[Tensor,Tensor,Tensor]:
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = torch.asin(2 * (w * y - z * x))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return roll, pitch, yaw



  
