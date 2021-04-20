import numpy as np
import heapq
import math
import pdb


# parameters for the pure pursuit controller
POSITION_THRESHOLD = 0.04
MIN_VELOCITY = 0.0
REF_VELOCITY = 0.7      # 0.8
LOW_VELOCITY = 0.15 * 0.7
MAX_VELOCITY = 1.0
MIN_STEERING = -2 * math.pi
MAX_STEERING = 2 * math.pi
GAIN = 10
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
    DP_HORIZON = 12
    NUM_DISCRETE_CONTROLS = 10
    SPATIAL_RESOLUTION = 0.01       # 0.004
    ANGLE_RESOLUTION = 0.1 * math.pi
    SAFETY_REWARD_THRESHOLD = -100
    SAFETY_FACTOR = 2.0
    NEIGHBOUR_HALF_SIZE = 0.5
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.last_steering = 0.0
        self.astar_controls = [(REF_VELOCITY, 0), (LOW_VELOCITY, 0), (0, -2.14 * math.pi), (0, 2.14 * math.pi)]
        self.dp_controls = [(MAX_VELOCITY, 0), (0, 2 * math.pi)]
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
        elif abs(self.last_steering) > 0.2:
            lookup_distance = FOLLOWING_DISTANCE * 0.5
        elif self.env._proximity_penalty2(self.env.cur_pos, self.env.cur_angle) < -0.01:
            lookup_distance = FOLLOWING_DISTANCE * 0.3
        else:
            lookup_distance = FOLLOWING_DISTANCE

        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent, then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _, curve_tangent = self.env.closest_curve_point(follow_point, self.env.cur_angle,
                                                                         return_atan=True)

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
        elif self.env._proximity_penalty2(self.env.cur_pos, self.env.cur_angle) < -0.01:
            velocity = REF_VELOCITY * 0.5
        else:
            velocity = MAX_VELOCITY

        # check failure states
        if self.pred_reward(velocity, steering, safety_factor=self.SAFETY_FACTOR) < self.SAFETY_REWARD_THRESHOLD:
            print(f'Failure state detected. Recovery starts.')
            # self.hybrid_astar_search(curve_point, math.atan(curve_tangent))
            self.dp_search(self.DP_HORIZON)
            return self.action_seq.pop(0)

        return velocity, steering

    def hybrid_astar_search(self, goal_pose, goal_angle):
        print(goal_pose, goal_angle)
        frontier, explored = PriorityQueue(), {}
        self.init_grid()
        init_pose, init_angle = self.env.cur_pos, self.env.cur_angle
        self.init_pose, self.init_angle = init_pose, init_angle
        print(f'initial pose and angle: {self.discretize_state((init_pose, init_angle))}\n'
              f'goal pose and angle {self.discretize_state((goal_pose, goal_angle))}\n')
        print(f'initial pose and angle: {((init_pose, init_angle))}\n'
              f'goal pose and angle {((goal_pose, goal_angle))}\n')

        # init frontier
        frontier.update((None, 0, None, ((init_pose[0], init_pose[-1]), init_angle)), 0)   # action, cost, parent, state

        # search loop
        while not frontier.isEmpty():
            node = frontier.pop()
            (action, cost, parent, curr_state) = node
            grid_coord = self.discretize_state(curr_state)

            if curr_state in explored:
                continue

            explored[curr_state] = (action, cost, parent)      # action, cost, parent

            # check goal
            if self.check_goal(curr_state, goal_pose, goal_angle):
                self.action_seq = backtrack(explored, curr_state)
                print('controls: ', self.action_seq)
                return

            # update grid cell state
            self.set_grid_cell_value(grid_coord, node)

            # enumerate children
            children = self.get_successors(node)
            for child in children:
                c_action, c_cost, c_parent, c_state = child
                c_grid_coord = self.discretize_state(c_state)

                if not self.check_within_grid_bound(c_grid_coord):
                    continue

                if (not self.is_grid_cell_empty(c_grid_coord)) and \
                        c_cost >= self.get_grid_cell_value(c_grid_coord)[1]:
                    continue

                estimated_cost = c_cost + self.manhattan_dist(c_state[0], goal_pose)
                frontier.update(child, estimated_cost)

        return NotImplementedError()

    def get_world_size(self):
        return self.env.grid_width, self.env.grid_height

    def init_grid(self):
        w, h = self.get_world_size()
        self.grid_size = (int(2 * self.NEIGHBOUR_HALF_SIZE * w / self.SPATIAL_RESOLUTION) + 1,
                          int(2 * self.NEIGHBOUR_HALF_SIZE * h / self.SPATIAL_RESOLUTION) + 1,
                          int(2 * math.pi / self.ANGLE_RESOLUTION) + 1)
        self.grid = np.zeros(self.grid_size).tolist()
        print(f'grid initialized, grid shape {self.grid_size}')

    def transform_x(self, x):
        return x - (self.init_pose[0] - self.NEIGHBOUR_HALF_SIZE)

    def transform_y(self, y):
        return y - (self.init_pose[-1] - self.NEIGHBOUR_HALF_SIZE)

    def discretize_x(self, x):
        return int(round(self.transform_x(x) / self.SPATIAL_RESOLUTION))

    def discretize_y(self, y):
        return int(round(self.transform_x(y) / self.SPATIAL_RESOLUTION))

    def discretize_angle(self, theta):
        while theta < 0:
            theta += 2 * math.pi
        return int(round(theta % (2 * math.pi) / self.ANGLE_RESOLUTION))

    def discretize_state(self, state):
        coords, angle = state
        x, y = coords[0], coords[-1]        # the coordinate might be (x, 0, z) or (x, z)
        x, y = self.discretize_x(x), self.discretize_y(y)
        angle = self.discretize_angle(angle)
        return x, y, angle

    def check_within_grid_bound(self, coord):
        a, b, c = coord
        return a < self.grid_size[0] and b < self.grid_size[1] \
               and c < self.grid_size[2] and self.grid[a][b][c] == 0

    def get_grid_cell_value(self, coord):
        a, b, c = coord
        return self.grid[a][b][c]

    def is_grid_cell_empty(self, coord):
        a, b, c = coord
        assert self.check_within_grid_bound(coord), \
            f'index {(a, b, c)} out of range of grid {self.grid_size}'
        return a < self.grid_size[0] and b < self.grid_size[1] \
               and c < self.grid_size[2] and self.grid[a][b][c] == 0

    def set_grid_cell_value(self, coord, node):
        print(f'coord {coord}')
        a, b, c = coord
        self.grid[a][b][c] = node

    def check_goal(self, state, goal_pose, goal_angle):
        x, y, angle = self.discretize_state(state)
        goal_x, goal_y, goal_angle = goal_pose[0], goal_pose[-1], goal_angle
        goal_x, goal_y, goal_angle = self.discretize_state(((goal_pose[0], goal_pose[-1]), goal_angle))
        return x == goal_x and y == goal_y and angle == goal_angle

    def get_discrete_controls(self, n):
        controls = []
        for i in range(n):
            v = (MAX_VELOCITY - MIN_VELOCITY) * i / n + MIN_VELOCITY
            for j in range(n):
                s = (MAX_STEERING - MIN_STEERING) * j / n + MIN_STEERING
                controls.append((v, s))
        return controls

    def manhattan_dist(self, coord_1, coord_2):
        return ((coord_1[0] - coord_2[0]) ** 2 + (coord_1[-1] - coord_2[-1]) ** 2) ** 0.5

    def get_successors(self, state):
        action, cost, parent, (pos, angle) = state       # action, cost, parent, state
        successors = []
        for v, s in self.astar_controls:
            new_state = self.pred_physics(v, s, pos, angle)
            new_pos, new_angle, new_speed = new_state
            step_cost = self.manhattan_dist(pos, new_pos)
            # reward = self.pred_reward(v, s, pos, angle)
            ## if reward <= -100: #or not self.env._valid_pose(new_pos, new_angle, safety_factor=1.5):
            #    continue
            successors.append(((v, s), cost + step_cost, (pos, angle),
                               ((new_pos[0], new_pos[-1]), new_angle)))
        return successors

    def pred_physics(self, v, s, pos, angle):
        action = self.env.convert_action_to_vels(np.array([v, s]))
        if pos is not None and len(pos) == 2:
            pos = np.array([pos[0], 0.0, pos[-1]])
        return self.env.pred_physics(action, pos, angle)

    def pred_reward(self, v, s, pos=None, angle=None, safety_factor=1.0):
        cur_pos, cur_angle, real_speed = self.pred_physics(v, s, pos, angle)
        if not self.env._valid_pose(cur_pos, cur_angle, safety_factor=safety_factor):
            return -1000
        else:
            return self.env.compute_reward(cur_pos, cur_angle, real_speed)

    def dp_search(self, horizon):
        # DP tables
        rewards, pose = [[] for i in range(horizon)], [[] for i in range(horizon)]
        successors, actions = {}, {}

        # DP
        for t in range(horizon):
            if t == 0:
                for i, (v, s) in enumerate(self.dp_controls):
                    rewards[t].append(self.pred_reward(v, s))
                    pose[t].append(self.pred_physics(v, s, None, None))
                    successors[(i, t)] = None
                    actions[(i, t)] = (v, s)
                continue

            for i, control in enumerate(rewards[t - 1]):
                pos, angle, _ = pose[t - 1][i]
                for j, (v, s) in enumerate(self.dp_controls):
                    rewards[t].append(rewards[t - 1][i] + self.pred_reward(v, s, pos, angle))
                    pose[t].append(self.pred_physics(v, s, pos, angle))
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
