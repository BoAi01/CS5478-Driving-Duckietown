import numpy as np
import heapq
import math


# parameters for the pure pursuit controller
POSITION_THRESHOLD = 0.04
MIN_VELOCITY = 0.0
REF_VELOCITY = 0.7      # 0.8
LOW_VELOCITY = 0.15 * 0.7
MAX_VELOCITY = 1.0
MIN_STEERING = -5.0
MAX_STEERING = 5.0
GAIN = 10      # 10
FOLLOWING_DISTANCE = 0.6  # 0.3


class PriorityQueue:
    REMOVED = '<removed-task>'

    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.count = 0

    def update(self, item, priority):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)
        self.count += 1

    def pop(self):
        while self.pq:
            (_, _, item) = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError("priority queue empty")

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def isEmpty(self):
        return len(self) == 0

    def __len__(self):
        return len(self.entry_finder)


def backtrack(explored, state):
    goal_actions = []
    while True:
        curr_val = explored.get(state)      # action, cost, parent
        state = curr_val[-1]
        if state is None:
            break
        goal_actions.append(curr_val[0])
    goal_actions.reverse()
    return goal_actions


class PurePursuitExpert:
    NUM_DISCRETE_CONTROLS = 10
    SPATIAL_RESOLUTION = 100
    ANGLE_RESOLUTION = 10
    SPEED_RESOLUTION = 5
    HORIZON = 14
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.last_steering = 0.0
        self.controls = self.get_discrete_controls(self.NUM_DISCRETE_CONTROLS)
        self.action_seq = []

    def predict(self, observation):  # we don't really care about the observation for this implementation
        # check planned actions
        if len(self.action_seq) > 0:
            return self.action_seq.pop(0)

        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        # pure pursuit
        iterations = 0

        if self.env.is_there_stop_sign(self.env.cur_pos):
            lookup_distance = FOLLOWING_DISTANCE * 0.05
        elif abs(self.last_steering) > 0.2 or self.env._proximity_penalty2(self.env.cur_pos, self.env.cur_angle) < -0.01:
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
        elif abs(steering) > 0.2 or self.env._proximity_penalty2(self.env.cur_pos, self.env.cur_angle) < -0.01:
            velocity = REF_VELOCITY
        else:
            velocity = MAX_VELOCITY

        # check failure states
        if self.pred_reward(velocity, steering, safety_factor=1.5) < -10:
            print(f'Failure state detected. Recovery starts. Horizon = {self.HORIZON}')
            # self.hybrid_astar_search(self.env.cur_pos, self.env.cur_angle, self.env.speed / 1.2)
            self.dp_search(self.HORIZON)
            return self.action_seq.pop(0)

        return velocity, steering


    def dp_search(self, horizon):
        # control space
        controls = [(MAX_VELOCITY, 0), (0, 2 * math.pi)]

        # DP tables
        rewards, pose = [[] for i in range(horizon)], [[] for i in range(horizon)]
        successors, actions = {}, {}

        # DP
        for t in range(horizon):
            if t == 0:
                for i, (v, s) in enumerate(controls):
                    rewards[t].append(self.pred_reward(v, s))
                    pose[t].append(self.pred_physics(v, s, None, None, None))
                    successors[(i, t)] = None
                    actions[(i, t)] = (v, s)
                continue

            for i, control in enumerate(rewards[t - 1]):
                pos, angle, speed = pose[t - 1][i]
                for j, (v, s) in enumerate(controls):
                    rewards[t].append(rewards[t - 1][i] + self.pred_reward(v, s, pos, angle, speed))
                    pose[t].append(self.pred_physics(v, s, pos, angle, speed))
                    successors[(len(rewards[t]) - 1, t)] = (i, t - 1)
                    actions[(len(rewards[t]) - 1, t)] = (v, s)

        # backtrack controls
        r, t = np.argmax(rewards[-1]), horizon - 1
        controls, record_rewards = [], []
        while True:
            controls.append(actions[(r, t)])
            record_rewards.append(rewards[t][r])
            if successors[(r, t)] is None:
                break
            r, t = successors[(r, t)]

        # publish
        controls.reverse()
        record_rewards.reverse()
        print(f'DP controls: {controls}')
        print(f'DP optimal average reward: {record_rewards[-1] / len(record_rewards)}')
        self.action_seq = controls

        return

    def hybrid_astar_search(self, pos, angle, speed):
        frontier = PriorityQueue()
        explored = {}
        self.grid = np.zeros((self.SPATIAL_RESOLUTION, self.SPATIAL_RESOLUTION,
                              self.ANGLE_RESOLUTION, self.SPEED_RESOLUTION * 2)).tolist()

        # action, cost, parent, state
        frontier.update((None, 0, None, ((pos[0], pos[-1]), angle, speed)), 0)
        while not frontier.isEmpty():
            node = frontier.pop()
            (action, cost, parent, curr_state) = node
            grid_coord = self.discrete_state(curr_state)

            if curr_state in explored:
                continue

            explored[curr_state] = (action, cost, parent)

            # check goal
            if self.check_goal(curr_state):
                self.action_seq = backtrack(explored, curr_state)
                print('action seq ', self.action_seq)
                return

            # update grid cell state
            self.set_grid_cell_value(grid_coord, node)

            # enumerate children
            children = self.get_successors(node)
            for child in children:
                c_action, c_cost, c_parent, c_state = child
                c_grid_coord = self.discrete_state(c_state)
                # check cost
                print(c_state, c_grid_coord)
                if (not self.is_grid_cell_empty(c_grid_coord)) and c_cost >= self.get_grid_cell(c_grid_coord)[1]:
                    continue

                esti_cost = c_cost
                print(child)
                frontier.update(child, esti_cost)

        import pdb
        pdb.set_trace()
        return NotImplementedError()

    def get_grid_cell(self, coord):
        a, b, c, d = coord
        return self.grid[a][b][c][d]

    def is_grid_cell_empty(self, coord):
        a, b, c, d = coord
        print(a, b, c, d)
        return self.grid[a][b][c][d] == 0

    def set_grid_cell_value(self, coord, node):
        a, b, c, d = coord
        self.grid[a][b][c][d] = node

    def check_goal(self, state):
        return self.pred_reward(1, 0, state[0], state[1], state[2]) > 0.6

    def discrete_state(self, state):
        pos, angle, speed = state
        dis_pos = (int(round(pos[0] * self.SPATIAL_RESOLUTION)),
                  int(round(pos[-1] * self.SPATIAL_RESOLUTION)))
        dis_angle = int(round(angle / (2 * math.pi / self.ANGLE_RESOLUTION)))
        dis_speed = int(round(speed * self.SPEED_RESOLUTION))
        return dis_pos[0], dis_pos[-1], dis_angle, dis_speed

    def get_discrete_controls(self, n):
        controls = []
        for i in range(n):
            v = (MAX_VELOCITY - MIN_VELOCITY) * i / n + MIN_VELOCITY
            for j in range(n):
                s = (MAX_STEERING - MIN_STEERING) * j / n + MIN_STEERING
                controls.append((v, s))
        return controls

    def get_successors(self, state):
        action, cost, parent, (pos, angle, speed) = state       # action, cost, parent, state
        successors = []
        for v, s in self.controls:
            new_state = self.pred_physics(v, s, pos, angle, speed)
            new_pos, new_angle, new_speed = new_state
            reward = self.pred_reward(v, s, pos, angle, speed)
            if reward <= -100:
                continue
            successors.append(((v, s), cost - reward, (pos, angle, speed),
                               ((new_pos[0], new_pos[-1]), new_angle, new_speed)))
        return successors

    def pred_physics(self, v, s, pos, angle, robot_speed):
        action = self.env.convert_action_to_vels(np.array([v, s]))
        if pos is not None and len(pos) == 2:
            pos = np.array([pos[0], 0.0, pos[-1]])
        return self.env.pred_physics(action, pos, angle, robot_speed)

    def pred_reward(self, v, s, pos=None, angle=None, robot_speed=None, safety_factor=1.0):
        cur_pos, cur_angle, real_speed = self.pred_physics(v, s, pos, angle, robot_speed)
        if not self.env._valid_pose(cur_pos, cur_angle, safety_factor=safety_factor):
            return -1000
        else:
            return self.env.compute_reward(cur_pos, cur_angle, real_speed)
