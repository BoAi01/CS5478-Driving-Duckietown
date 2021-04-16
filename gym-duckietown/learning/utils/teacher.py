import numpy as np


# parameters for the pure pursuit controller
POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7      # 0.8
LOW_VELOCITY = 0.15 * 0.7
MAX_VELOCITY = 1.0
GAIN = 10      # 10
FOLLOWING_DISTANCE = 0.6  # 0.3


class PurePursuitExpert:
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.last_steering = 0.0

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        # pure pursuit
        iterations = 0

        if self.env.is_there_stop_sign(self.env.cur_pos):
            lookup_distance = FOLLOWING_DISTANCE * 0.05
        elif abs(self.last_steering) > 0.2:
            lookup_distance = FOLLOWING_DISTANCE * 0.5
        else:
            lookup_distance = FOLLOWING_DISTANCE

        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)
            # If we have a valid point on the curve, stop
            if curve_point is not None:
                 break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos

        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = GAIN * -dot
        self.last_steering = steering

        if self.env.is_there_stop_sign(self.env.cur_pos):
            velocity = LOW_VELOCITY
        elif abs(steering) > 0.2:
            velocity = REF_VELOCITY
        else:
            velocity = MAX_VELOCITY

        # check failure case
        if self.pred_reward(velocity, steering) < -10:
            velocity = 0.0
            steering = 0.5

        return velocity, steering





        """# check stop sign
        if self.env.is_there_stop_sign(self.env.cur_pos):
            return LOW_VELOCITY, steering

        velocity = REF_VELOCITY

        # check high loss
        if True or self.pred_reward(velocity, steering) < -5:
            return velocity, steering

        # DP to search optimal velocity
        max_velocity = MAX_VELOCITY
        n = 10
        horizon = 5
        table = np.zeros((n + 1, horizon))
        pose = np.zeros((n + 1, horizon)).tolist()
        last_action = {}
        for t in range(horizon):
            for r in range(n + 1):
                v = max_velocity * r / n
                if t == 0:
                    table[r][t] = self.pred_reward(v, steering)
                    pose[r][t] = self.pred_physics(v, steering, None, None, None)
                    last_action[(r, t)] = None
                    continue

                max_value = -float('inf')
                for r2 in range(n + 1):
                    pos, angle, robot_speed = pose[r2][t - 1]
                    value = table[r2, t - 1] + self.pred_reward(v, steering, pos, angle, robot_speed)
                    if value > max_value:
                        max_value = value
                        pose[r][t] = self.pred_physics(v, steering, pos, angle, robot_speed)
                        last_action[(r, t)] = (r2, t - 1)
                table[r][t] = max_value

        r = np.argmax(table[:, -1])
        t = horizon - 1
        while last_action[(r, t)] is not None:
            r, t = last_action[(r, t)]

        velocity = max_velocity * r / n"""

        """velocity = 1.0

        # adjust velocity
        n = 10
        max_v, max_reward = velocity, -float('inf')
        rewards = []
        for i in range(n + 1):
            curr_v = i / n * velocity
            pred_reward = self.pred_reward(curr_v, steering)
            rewards.append(pred_reward)
            if pred_reward > max_reward:
                max_v = curr_v
                max_reward = pred_reward
        velocity = max_v
        print(f'velocity {rewards}')
        # return velocity, steering

        # adjust steering
        max_s, max_reward = velocity, -float('inf')
        rewards = []
        for i in range(2 * (n + 1)):
            curr_s = i / n * steering - steering
            pred_reward = self.pred_reward(velocity, curr_s)
            rewards.append(pred_reward)
            if pred_reward > max_reward:
                max_s = curr_s
                max_reward = pred_reward
        steering = max_s
        print(f'steering {rewards}')"""

        # velocity = velocity if abs(steering) < 4 else velocity * 0.7
        # print(velocity, steering)
        import pdb
        # pdb.set_trace()

        return velocity, steering

    def pred_physics(self, v, s, pos, angle, robot_speed):
        action = np.array([v, s])
        return self.env.pred_physics(action, pos, angle, robot_speed)

    def pred_reward(self, v, s, pos=None, angle=None, robot_speed=None):
        cur_pos, cur_angle, real_speed = self.pred_physics(v, s, pos, angle, robot_speed)
        if not self.env._valid_pose(cur_pos, cur_angle):
            return -1000
        else:
            return self.env.compute_reward(cur_pos, cur_angle, real_speed)
