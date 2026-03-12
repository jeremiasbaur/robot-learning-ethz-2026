import numpy as np

# ## Deliverables
# 1. **Video:** Video (.mp4) of the robot moving between the waypoints. **The video length must be less than 2 minutes** including robot motion and theoretical questions.
# 2. **Code:** Your code with filled in TODOs in `exercises/ex2.py`.
# 3. **Theoretical questions**. The video must include your answers to the theoretical questions.


# To get a feeling for the choice of the PID gains, you will analyze how their choice influences the behavior of the waypoint tracking. 
# Test different settings of the gains to be able to answer the following:
# 1. If you keep increasing $K_P$, what issue arises when tracking the waypoints?
# If the error is large and we have a high K_P, then a strong correction signal is applied and there is the risk of an overshoot
# and oscillations around the target.

# 2. How does $K_D$ mitigate the effect you saw above when increasing $K_P$?
# The K_D term penalizes changes of the error and can be seen as damping.
# If we have a lot of harsh movements due to a high K_P term, we get a high derivative which is penalized by K_D.

# 3. In what scenarios is a non-zero $K_I$ needed for the controller to perform well?
# If there is a steady-state error (due to e.g. gravity) that we want to get rid of, then the Integral term will counter-act it.

def generate_quintic_spline_waypoints(start, end, num_points):

    """
    TODO:

    Steps:
    1. Generate `num_points` linearly spaced time steps `s` between 0 and 1.
    2. Apply the quintic time scaling polynomial function which can be found in the slides to get `f_s`.
    3. Interpolate between `start` and `end` using `start + (end - start) * f_s`.
    
    Args:
        start (np.ndarray): Starting waypoint.
        end (np.ndarray): Ending waypoint.
        num_points (int): Number of points in the trajectory.
        
    Returns:
        np.ndarray: Generated waypoints.
    """

    s = np.linspace(0, 1, num_points)
    f_s = 10*s**3 - 15*s**4 + 6*s**5

    waypoints = start + (end-start) * f_s[:, np.newaxis]
    
    return waypoints


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0, Kd=0.02):
    """
    TODO:
    Compute the PID control signal based on the tracking error history.
    
    Steps:
    1. The Proportional (P) term is the most recent error.
    2. The Integral (I) term is the sum of all past errors, multiplied by the simulation timestep.
    3. The Derivative (D) term is the rate of change of the error (difference between the last two errors divided by the timestep).
       If there is only one error in history, the D term should be zero.
    4. Compute the final control signal: Kp * P + Ki * I + Kd * D.
    
    Args:
        tracking_error_history (np.ndarray): History of tracking errors.
        timestep (float): Simulation timestep.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        
    Returns:
        np.ndarray: Control signal.
    """
    if len(tracking_error_history) == 0:
        return 0
    
    integral = np.sum(tracking_error_history)
    d_term = 0 if len(tracking_error_history) == 1 else (tracking_error_history[-1] - tracking_error_history[-2])
    return Kp * tracking_error_history[-1] + Ki * integral * timestep + Kd * d_term / timestep
            