from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as image
import matplotlib.widgets as widgets
import matplotlib.patches as mpatches


class Simulation:
    def __init__(
        self,
        pop_size: int = 5000,
        init_case: int = 1,
        threshold: np.float64 = 2.0,
        map_size: np.float64 = 100,
        hospital_beds_ratio: float = .003,
        infection_time: np.uint64 = 20,
        p_infect: float = 0.3,
        p_recover: float = 0.7
    ):
        self.time: int = 0

        self.pop_size = pop_size
        self.map_size = map_size
        self.threshold = threshold
        self.hospital_beds_ratio = hospital_beds_ratio
        self.p_infect: float = p_infect
        self.p_recover: float = p_recover
        self.p_die: float = 1 - p_recover
        self.infection_time: np.uint64 = infection_time
        self.x = np.random.uniform(
            low=0.0, high=map_size, size=(pop_size, 1))
        self.y = np.random.uniform(
            low=0.0, high=map_size, size=(pop_size, 1))

        self.health_code = {'healthy': 0, 'infected': 1,
                            'recovered': 2, 'deceased': 3}
        # self.health = np.zeros((pop_size, 1), dtype=np.uint64)
        self.health: np.ndarray = \
            np.full(
                shape=(self.pop_size, 1),
                fill_value=self.health_code['healthy'],
                dtype=np.ubyte
            )
        self.health[0:init_case] = self.health_code['infected']
        # index_array = np.random.choice(pop_size, init_case, replace=False)
        # self.health[index_array] = self.health_code['infected']

        self.infected_duration: np.ndarray = \
            np.zeros(shape=(self.pop_size,), dtype=np.uint64)
        self.infected_duration[0:init_case] = 1

        self.infected: List[int] = []
        self.recovered: List[int] = []
        self.dead: List[int] = []
        self.new_cases: List[int] = []

        self.fig = plt.figure(figsize=(16, 9))

        self.fig.legend(handles=[
            mpatches.Patch(color='blue', label='Healthy'),
            mpatches.Patch(color='red', label='Infected'),
            mpatches.Patch(color='green', label='Recovered'),
            mpatches.Patch(color='black', label='Deceased'),
        ], loc='lower left')

        self.grid = self.fig.add_gridspec(nrows=3, ncols=4)
        self.ax = self.fig.add_subplot(self.grid[0:, 1:])
        self.stats_ax = self.fig.add_subplot(self.grid[0, 0])
        self.traj_ax = self.fig.add_subplot(self.grid[1, 0])
        self.check_ax = self.fig.add_subplot(self.grid[2, 0], frame_on=False)
        # Add labels
        self.stats_ax.set_xlabel('Time')
        self.stats_ax.set_ylabel('Cases')
        self.traj_ax.set_xlabel('Log(Total Cases)')
        self.traj_ax.set_ylabel('Log(New Cases)')

        self.ani = animation.FuncAnimation(
            self.fig, self.update_map, init_func=self.init_map)

        self.stats_ani = animation.FuncAnimation(
            self.fig, self.update_stats)

        # self.ax.imshow(
        #     image.imread('map.png'),
        #     extent=[0, 100, 0, 100], aspect='auto')

        self.paused: bool = False
        self.log_scale: bool = False
        self.p_movement: float = 1.0
        self.checks = widgets.CheckButtons(self.check_ax, [
            'Use Log Scale',
            'Total Social Distancing',
            'Partial Social Distancing',
            'Paused',
        ])

        self.checks.on_clicked(self.checkbox_handler)

    def update_location(self, step_mean: np.float64 = 2.0):
        move = np.random.normal(
            scale=np.sqrt(step_mean/2),
            size=(self.pop_size, 1))

        move[
            (self.health == self.health_code['infected']) |
            (np.random.binomial(1, self.p_movement, size=(self.pop_size,)) == 0)
        ] = np.array([0, 0])

        self.x += move
        self.y += move
        self.x = np.clip(self.x, 0.0, self.map_size)
        self.y = np.clip(self.y, 0.0, self.map_size)

    def find_p_at_risk(self):
        positions = np.c_[self.x, self.y]
        distVectors: np.ndarray = positions - \
            positions.reshape((self.pop_size, 1, 2))
        distMat: np.ndarray = np.sqrt(np.sum(distVectors**2, axis=2))

        possibilityMat: np.ndarray = \
            (distMat <= self.threshold) * \
            (self.health == self.health_code['infected']) * \
            (self.health == self.health_code['healthy']).reshape(
                (self.pop_size, 1))

        ppl_at_risk = np.sum(possibilityMat, axis=1)

        # xB = self.x.T
        # dx = self.x-xB
        # yB = self.y.T
        # dy = self.y-yB
        # distance = (dx**2 + dy**2)**0.5
        # too_close_ppl = (distance <= self.threshold)
        # healthy_rows_and_too_close = too_close_ppl * \
        #     ((self.health == self.health_code['healthy']))
        # healthy_rows_and_infected_cols_too_close = healthy_rows_and_too_close * \
        #     ((self.health == self.health_code['infected']).T)
        # ppl_we_care_about = healthy_rows_and_too_close * \
        #     healthy_rows_and_infected_cols_too_close
        # interactions_we_care_about = too_close_ppl * ppl_we_care_about
        # ppl_at_risk = np.sum(interactions_we_care_about,
        #                      axis=1, keepdims=True)
        # print(distance)
        return ppl_at_risk

    def update_status(self):
        ppl_at_risk = self.find_p_at_risk()
        self.health = ppl_at_risk
        self.health[ppl_at_risk > 0] = self.health_code['infected']
        health = self.health.T
        self.infected_duration[health[0] == self.health_code['infected']] += 1
        infection_done = self.infected_duration >= self.infection_time

        self.health[infection_done] = self.health_code['deceased']
        self.health[np.random.binomial(
            infection_done, self.p_recover) > 0] = self.health_code['recovered']

        self.infected_duration[infection_done] = 0

    def health_colors(self):
        color = np.vectorize({
            self.health_code['healthy']: 'blue',
            self.health_code['infected']: 'red',
            self.health_code['recovered']: 'green',
            self.health_code['deceased']: 'black',
        }.get)(self.health).T
        return color[0]

    def init_map(self):
        # self.ax.imshow(
        #     image.imread('map.png'),
        #     extent=[0, self.map_size, 0, self.map_size],
        #     aspect='auto')

        self.colors = self.health_colors()
        self.scatter = self.ax.scatter(
            x=self.x,
            y=self.y,
            # s=10,
            c=self.colors,
        )
        return self.scatter

    def update_map(self, _):
        if not self.paused:
            self.update_location()
            self.update_status()
            self.scatter.set_facecolor(self.health_colors())
            self.positions = np.c_[self.x, self.y]
            self.scatter.set_offsets(self.positions)
            return self.scatter

    def update_stats(self, _):
        if not self.paused:
            # Compute the statistics
            self.time += 1
            self.infected.append(np.count_nonzero(
                self.health == self.health_code['infected']))
            self.recovered.append(np.count_nonzero(
                self.health == self.health_code['recovered']))
            self.dead.append(np.count_nonzero(
                self.health == self.health_code['deceased']))

            cum_infected = self.infected
            cum_recovered = list(map(sum, zip(cum_infected, self.recovered)))
            cum_dead = list(map(sum, zip(cum_recovered, self.dead)))

            self.new_cases.append(
                cum_dead[-1] - (cum_dead[-2] if len(cum_dead) >= 2 else 0))

            self.stats_ax.clear()
            self.stats_ax.axhline(
                y=self.hospital_beds_ratio * self.pop_size, color='red', linestyle='--')
            self.stats_ax.set_yscale('log' if self.log_scale else 'linear')
            self.stats_ax.fill_between(
                range(self.time), cum_infected, 0, color='red')
            self.stats_ax.fill_between(
                range(self.time), cum_recovered, cum_infected, color='green')
            self.stats_ax.fill_between(
                range(self.time), cum_dead, cum_recovered, color='black')

            self.traj_ax.set_xscale('log')
            self.traj_ax.set_yscale('log')
            self.traj_ax.plot(cum_dead, self.new_cases, color='red')

    def checkbox_handler(self, option):
        # Get the state of the buttons
        state: Tuple[bool, bool, bool, bool] = self.checks.get_status()
        # Set our model accordingly
        self.paused = state[3]
        self.log_scale = state[0]
        self.p_movement = 0.0 if state[1] else 0.6 if state[2] else 1.0


sim = Simulation()
plt.show()
# sim.update_location()
