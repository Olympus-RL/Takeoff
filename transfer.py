import torch
if __name__ == '__main__':
    cfg = torch.load('runs/HighJump/HighJump1.pth')
    model = cfg['model']
    print(cfg.keys())
    print(model.keys()) 
    optim = cfg['optimizer']
    scaler = cfg['scaler']    
    frame = cfg['frame']
    epoch = cfg['epoch']
    last_mean_r = cfg['last_mean_rewards']
    env_state = cfg['env_state']

    print(
        type(optim)
    )
    print(
        type(scaler)
    )
    print(
        type(frame)
    )
    print(
        type(epoch)
    )
    print(
        type(last_mean_r)
    )
  

    print(optim.keys())
    print(scaler.keys())
    print((optim['state'][1].keys()))