# Nice to have type hinting
from typing import List, Tuple

# Needed for efficient distance calculation
import numpy as np

# Needed for plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.widgets as widgets
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


class Simulation:
    def __init__(
        self,
        pop_size: int = 5000,
        init_case: int = 1,
        threshold: np.float64 = 2.0,
        map_size: np.float64 = 100,
        hospital_beds_ratio: float = 0.03,
        infection_time: np.uint64 = 20,
        p_infect: float = 0.3,
        p_recover: float = 0.7
    ):
        # Simulation starts at time t=0
        self.time: int = 0

        # Keep track of the population size for this simulation
        self.pop_size = pop_size
        self.map_size = map_size
        self.threshold = threshold
        self.hospital_beds_ratio = hospital_beds_ratio
        self.p_infect: float = p_infect
        self.p_recover: float = p_recover
        self.p_die: float = 1 - p_recover
        self.infection_time: np.uint64 = infection_time

        # Position and health of everyone in the population
        self.x = np.random.uniform(
            low=0.0, high=map_size, size=(pop_size, 1))
        self.y = np.random.uniform(
            low=0.0, high=map_size, size=(pop_size, 1))
        self.positions = np.c_[self.x, self.y]
        self.health_code = {'healthy': 0, 'infected': 1,
                            'recovered': 2, 'deceased': 3}
        self.health: np.ndarray = \
            np.full(
                shape=(self.pop_size,),
                fill_value=self.health_code['healthy'],
                dtype=np.ubyte
            )
        self.infected_duration: np.ndarray = \
            np.zeros(shape=(self.pop_size,), dtype=np.uint64)

        #  Note that we don't have to randomly draw from `self.healths`. The
        #  positions are all generated in the same way, and they're all
        #  indistinguishable from each other.
        self.health[0:init_case] = self.health_code['infected']
        self.infected_duration[0:init_case] = 1

        # Keep track of the statistics too
        self.infected: List[int] = []
        self.recovered: List[int] = []
        self.dead: List[int] = []
        self.new_cases: List[int] = []

        # Create the figure for plotting
        self.fig = plt.figure(figsize=(16, 9))

        # Legend
        # Keep it on the whole figure since the colors apply to everything
        self.fig.legend(handles=[
            mpatches.Patch(color='blue', label='Healthy'),
            mpatches.Patch(color='red', label='Infected'),
            mpatches.Patch(color='orange', label='Recovered'),
            mpatches.Patch(color='black', label='Deceased'),
        ], loc='lower left')

        self.fig.legend(handles=[mlines.Line2D([], [], color='darkblue', marker='*',
                                               markersize=15, label='All-In Eduspace')
                                 ], loc='upper center')

        # Add a grid for layout
        self.grid = self.fig.add_gridspec(nrows=80, ncols=100)

        # Add the map, image and statistics
        self.ax = self.fig.add_subplot(self.grid[0:, 40:])
        self.img = mpimg.imread('v1/img/map.png')
        self.ax.imshow(
            self.img,
            extent=[0, self.map_size, 0, self.map_size],
            aspect='auto')
        self.stats_ax = self.fig.add_subplot(self.grid[0:12, 0:35])
        self.case_ax = self.fig.add_subplot(self.grid[22:33, 0:35])
        self.recovered_ax = self.fig.add_subplot(self.grid[42:55, 0:13])
        self.deceased_ax = self.fig.add_subplot(self.grid[42:55, 22:35])
        self.check_ax = self.fig.add_subplot(
            self.grid[63:, 10:35], frame_on=False)

        # Add labels
        self.stats_ax.set_xlabel('Time')
        self.stats_ax.set_ylabel('Cases')
        self.case_ax.set_xlabel('Total Cases')
        self.case_ax.set_ylabel('New Cases')
        self.recovered_ax.set_xlabel('Time')
        self.recovered_ax.set_ylabel('Recovered')
        self.deceased_ax.set_xlabel('Time')
        self.deceased_ax.set_ylabel('Deceased')

        # Call for updates
        self.ani = animation.FuncAnimation(
            self.fig, self.update_map, init_func=self.init_map)
        self.stats_ani = animation.FuncAnimation(
            self.fig, self.update_stats)

        # Check buttons and simulation options
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
            size=(self.pop_size, 2))

        # Dead people don't move
        move[
            (self.health == self.health_code['deceased']) |
            (np.random.binomial(1, self.p_movement, size=(self.pop_size,)) == 0)
        ] = np.array([0, 0])

        # Update the positions
        self.positions += move
        self.positions = np.clip(self.positions, 0.0, self.map_size)

    def find_p_at_risk(self):
        # Compute the distance between everyone
        distVectors: np.ndarray = self.positions - \
            self.positions.reshape((self.pop_size, 1, 2))
        distMat: np.ndarray = np.sqrt(np.sum(distVectors**2, axis=2))

        # Figure out whether a person can be infected by another.
        possibilityMat: np.ndarray = \
            (distMat <= self.threshold) * \
            (self.health == self.health_code['infected']) * \
            (self.health == self.health_code['healthy']).reshape(
                (self.pop_size, 1))

        # Sum up how many we can get infected
        ppl_at_risk = np.sum(possibilityMat, axis=1)
        return ppl_at_risk

    def update_status(self):
        # Compute who becomes infected at the current time
        self.health[np.random.binomial(
            self.find_p_at_risk(), self.p_infect) > 0] = self.health_code['infected']

        # Compute who has been infected for a long time
        self.infected_duration[self.health ==
                               self.health_code['infected']] += 1
        infection_done = self.infected_duration >= self.infection_time

        # Figure out who lives and who dies
        # Recovery happens with probability `p_recover`
        self.health[infection_done] = self.health_code['deceased']
        self.health[np.random.binomial(
            infection_done, self.p_recover) > 0] = self.health_code['recovered']

        # If they are recovered or deceased, we don't want to reroll their status
        self.infected_duration[infection_done] = 0

    def health_colors(self):
        color = np.vectorize({
            self.health_code['healthy']: 'blue',
            self.health_code['infected']: 'red',
            self.health_code['recovered']: 'orange',
            self.health_code['deceased']: 'black',
        }.get)(self.health)
        return color

    def init_map(self):
        self.scatter = self.ax.scatter(
            x=self.x,
            y=self.y,
            s=5,
            c=self.health_colors(),
        )
        return self.scatter

    def update_map(self, _):
        if not self.paused:
            self.update_location()
            self.update_status()
            self.scatter.set_facecolor(self.health_colors())
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

            # self.stats_ax.clear()
            self.stats_ax.axhline(
                y=self.hospital_beds_ratio * self.pop_size, color='red', linestyle='--')
            self.stats_ax.set_yscale('log' if self.log_scale else 'linear')

            self.stats_ax.fill_between(
                range(self.time), cum_infected, 0, color='red')
            self.stats_ax.fill_between(
                range(self.time), cum_recovered, cum_infected, color='orange')
            self.stats_ax.fill_between(
                range(self.time), cum_dead, cum_recovered, color='black')

            self.case_ax.set_xscale('log' if self.log_scale else 'linear')
            self.case_ax.set_yscale('log' if self.log_scale else 'linear')
            self.case_ax.plot(cum_dead, self.new_cases, color='red')

            self.recovered_ax.set_yscale('linear')
            self.recovered_ax.plot(
                range(self.time), self.recovered, color='orange')

            self.deceased_ax.set_yscale('linear')
            self.deceased_ax.plot(
                range(self.time), self.dead, color='black')

    def checkbox_handler(self, option):
        # Get the state of the buttons
        state: Tuple[bool, bool, bool, bool] = self.checks.get_status()
        # Set our model accordingly
        self.paused = state[3]
        self.log_scale = state[0]
        self.p_movement = 0.0 if state[1] else 0.6 if state[2] else 1.0


sim = Simulation()
plt.show()
# sim.health_colors()
