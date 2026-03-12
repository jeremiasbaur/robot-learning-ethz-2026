import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from __init__ import *
from env.so100_tracking_env import SO100TrackEnv
from exercises.ex3 import reset_robot, reset_target_position


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on SO100 tracking")
    parser.add_argument("--load_run", type=str, default="1",
                        help="training id")
    parser.add_argument("--checkpoint", type=str, default="500",
                        help="checkpoint id")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")
    return parser.parse_args()

def reset_env(model, data):
    data.qpos[:] = reset_robot(env.default_qpos)
    data.mocap_pos[0] = reset_target_position(data.body("Base").xpos.copy())
     
def policy_callback(model, data):
    step_count = getattr(policy_callback, "step_count", 0)

    if step_count == 0:
        reset_env(model, data)
    elif step_count % (play_episode_length * env.ctrl_decimation) == 0:
        ee_tracking_error = np.linalg.norm(data.site("ee_site").xpos - data.mocap_pos[0])
        policy_callback.total_ee_tracking_errors.append(ee_tracking_error)
        print(f"Final EE tracking error: {ee_tracking_error:.4f}")
        #print(f"Has converged: {policy_callback.converged}, Final convergence time: {(policy_callback.convergence_time * 0.002 if policy_callback.converged else (step_count-policy_callback.last_reset_time)*0.002)}")
        reset_env(model, data)
        policy_callback.total_convergence_times.append((policy_callback.convergence_time * 0.002 if policy_callback.converged else (step_count-policy_callback.last_reset_time)*0.002))
        policy_callback.converged = False
        policy_callback.convergence_time = 0
        policy_callback.last_reset_time = policy_callback.step_count
        

    elif step_count % env.ctrl_decimation == 0:
        obs = env._get_obs()
        ee_tracking_error = np.linalg.norm(data.site("ee_site").xpos - data.mocap_pos[0])
        if ee_tracking_error <= 0.01 and not policy_callback.converged:
            policy_callback.converged = True
            policy_callback.convergence_time = policy_callback.step_count - policy_callback.last_reset_time
        elif ee_tracking_error > 0.01 and policy_callback.converged:
            policy_callback.converged = False

        action, _states = rl_model.predict(obs, deterministic=True)
        data.ctrl[:] = env._process_action(action)
    policy_callback.step_count = step_count + 1


if __name__ == "__main__":
    args = parse_args()
    policy_path = EXP_DIR / f"so100_tracking_{args.load_run}" / f"model_{args.checkpoint}.zip" 
    
    env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
    max_num_episodes = 10
    play_episode_length_s = 2
    play_episode_length = int(play_episode_length_s / env.ctrl_timestep)
    print(play_episode_length, env.ctrl_timestep)
    policy_callback.total_ee_tracking_errors = []
    policy_callback.total_convergence_times = []
    policy_callback.converged = False
    policy_callback.convergence_time = 0
    policy_callback.last_reset_time = 0

    print(f"Loading model from {policy_path}...")
    rl_model = PPO.load(policy_path, device=args.device)

    mujoco.set_mjcb_control(policy_callback)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and policy_callback.step_count <= max_num_episodes * play_episode_length * env.ctrl_decimation:
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(env.model.opt.timestep)
    mujoco.set_mjcb_control(None)

    avg_ee_tracking_error = np.mean(policy_callback.total_ee_tracking_errors)
    avg_convergence_times = np.mean(policy_callback.total_convergence_times)
    print(f"Average final EE tracking error: {avg_ee_tracking_error:.4f}")
    #print(f"Average convergence time: {avg_convergence_times:.4f}")