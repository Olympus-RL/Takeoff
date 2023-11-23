
import torch
from torch import Tensor

_motor_cutoff_speed = 350/30*torch.pi #self._task_cfg["env"]["control"]["motorCutoffFreq"]
_motor_degration_speed = 225/30*torch.pi #self._task_cfg["env"]["control"]["motorDegradationFreq"]
max_torque = 24.8
def _get_torque_limits(joint_velocities: Tensor) -> Tensor:
    a = -max_torque/(_motor_cutoff_speed-_motor_degration_speed)
    b = -a*_motor_cutoff_speed
    return (a*joint_velocities.abs() + b).clamp(min=0,max=max_torque)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    x = torch.linspace(-500/30*torch.pi, 500/30*torch.pi, 1000)
    y = _get_torque_limits(x).numpy()
    plt.plot(x.numpy()*30/torch.pi,y)
    plt.savefig("torque_limits.png")
