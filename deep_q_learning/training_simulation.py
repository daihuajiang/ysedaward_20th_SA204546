import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
#八勢路口紅綠燈階段編號
PHASE_J1_main_GREEN = 0  
PHASE_J1_main_YELLOW = 1
PHASE_J1_mior1_GREEN = 2
PHASE_J1_mior1_YELLOW = 3
PHASE_J1_mior2_GREEN = 4
PHASE_J1_mior2_YELLOW = 5
#紅樹林捷運站紅綠燈階段編號
PHASE_J0_GREEN = 0
PHASE_J0_YELLOW = 1
PHASE_J0_RED = 2
#台2線和台2乙線紅綠燈階段編號
PHASE_J15_1_GREEN = 0
PHASE_J15_1_YELLOW = 1
PHASE_J15_2_GREEN = 2
PHASE_J15_2_YELLOW = 3
"""
action:
0: 八勢路口紅綠燈階段0
1: 八勢路口紅綠燈階段2
2: 台2線和台2乙線紅綠燈階段0
3: 台2線和台2乙線紅綠燈階段1
4: action 0+2
5: action 0+3
6: action 1+2
7: action 1+3
"""
class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, red_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._red_duration = red_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._episode_E0_mean_speed_store = []
        self._episode_nE0_mean_speed_store = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._sum_E0_mean_speed = 0
        self._sum_nE0_mean_speed = 0
        #self._same_action_counter = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)
            while(old_action == 1 or old_action == 6 or old_action == 7):
                if action == 1 or action == 6 or action == 7:
                    action = self._choose_action(current_state, epsilon)
                else:
                    break
            """
            if old_action == action and action == 0:
                self._same_action_counter = self._same_action_counter + 1
            else:
                self._same_action_counter = 0
            
            if self._same_action_counter >= 23:
                while(action == 0):
                    #action = self._choose_action(current_state, epsilon)
                    action = 1
            """

            # if the chosen phase is different from the last phase, activate the yellow phase
            if old_action != action:
                if action == 0:
                    self._set_phase("J1", PHASE_J1_main_YELLOW)
                    self._simulate(self._yellow_duration)
                """
                elif action == 2:
                    self._set_phase("J15", PHASE_J15_2_YELLOW)
                    self._simulate(self._yellow_duration)
                
                elif action == 3:
                    self._set_phase("J15", PHASE_J15_1_YELLOW)
                    self._simulate(self._yellow_duration)

                elif action == 4:
                    self._set_phase("J1", PHASE_J1_main_YELLOW)
                    self._set_phase("J15", PHASE_J15_2_YELLOW)
                    self._simulate(self._yellow_duration)

                elif action == 5:
                    self._set_phase("J1", PHASE_J1_main_YELLOW)
                    self._set_phase("J15", PHASE_J15_1_YELLOW)
                    self._simulate(self._yellow_duration)
                """
            # execute the phase selected before
            if action == 0:
                self._set_phase("J1", PHASE_J1_main_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._green_duration)

            elif action == 1:
                self._set_phase("J1", PHASE_J1_main_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._green_duration-3)
                self._set_phase("J1", PHASE_J1_main_YELLOW)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior1_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._red_duration)
                self._set_phase("J1", PHASE_J1_main_YELLOW)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior2_GREEN)
                self._set_phase("J0", PHASE_J0_YELLOW)
                self._set_phase("J18", PHASE_J0_YELLOW)
                self._set_phase("J22", PHASE_J0_YELLOW)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior2_GREEN)
                self._set_phase("J0", PHASE_J0_RED)
                self._set_phase("J18", PHASE_J0_RED)
                self._set_phase("J22", PHASE_J0_RED)
                self._simulate(self._red_duration-3)
                self._set_phase("J1", PHASE_J1_mior2_YELLOW)
                self._set_phase("J0", PHASE_J0_RED)
                self._set_phase("J18", PHASE_J0_RED)
                self._set_phase("J22", PHASE_J0_RED)
                self._simulate(self._yellow_duration)
            """
            elif action == 2:
                self._set_phase("J15", PHASE_J15_1_GREEN)
                self._simulate(self._green_duration)

            elif action == 3:
                self._set_phase("J15", PHASE_J15_2_GREEN)
                self._simulate(self._green_duration)

            elif action == 4:
                self._set_phase("J1", PHASE_J1_main_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._set_phase("J15", PHASE_J15_1_GREEN)
                self._simulate(self._green_duration)

            elif action == 5:
                self._set_phase("J1", PHASE_J1_main_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._set_phase("J15", PHASE_J15_2_GREEN)
                self._simulate(self._green_duration)

            elif action == 6:
                self._set_phase("J1", PHASE_J1_main_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._set_phase("J15", PHASE_J15_1_GREEN)
                self._simulate(self._green_duration-3)
                self._set_phase("J1", PHASE_J1_main_YELLOW)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._set_phase("J15", PHASE_J15_1_YELLOW)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior1_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._red_duration)
                self._set_phase("J1", PHASE_21_main_YELLOW)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior2_GREEN)
                self._set_phase("J0", PHASE_J0_YELLOW)
                self._set_phase("J18", PHASE_J0_YELLOW)
                self._set_phase("J22", PHASE_J0_YELLOW)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior2_GREEN)
                self._set_phase("J0", PHASE_J0_RED)
                self._set_phase("J18", PHASE_J0_RED)
                self._set_phase("J22", PHASE_J0_RED)
                self._simulate(self._red_duration-3)
                self._set_phase("J1", PHASE_J1_mior2_YELLOW)
                self._set_phase("J0", PHASE_J0_RED)
                self._set_phase("J18", PHASE_J0_RED)
                self._set_phase("J22", PHASE_J0_RED)
                self._simulate(self._yellow_duration)

            elif action == 7:
                self._set_phase("J1", PHASE_J1_main_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN-)
                self._set_phase("J15", PHASE_J15_2_GREEN)
                self._simulate(self._green_duration-3)
                self._set_phase("J1", PHASE_J1_main_YELLOW)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._set_phase("J15", PHASE_J15_2_YELLOW)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior1_GREEN)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._red_duration)
                self._set_phase("J1", PHASE_21_main_YELLOW)
                self._set_phase("J0", PHASE_J0_GREEN)
                self._set_phase("J18", PHASE_J0_GREEN)
                self._set_phase("J22", PHASE_J0_GREEN)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior2_GREEN)
                self._set_phase("J0", PHASE_J0_YELLOW)
                self._set_phase("J18", PHASE_J0_YELLOW)
                self._set_phase("J22", PHASE_J0_YELLOW)
                self._simulate(self._yellow_duration)
                self._set_phase("J1", PHASE_J1_mior2_GREEN)
                self._set_phase("J0", PHASE_J0_RED)
                self._set_phase("J18", PHASE_J0_RED)
                self._set_phase("J22", PHASE_J0_RED)
                self._simulate(self._red_duration-3)
                self._set_phase("J1", PHASE_J1_mior2_YELLOW)
                self._set_phase("J0", PHASE_J0_RED)
                self._set_phase("J18", PHASE_J0_RED)
                self._set_phase("J22", PHASE_J0_RED)
                self._simulate(self._yellow_duration)
            """
            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            E0_mean_speed = self._get_mean_speed("E0")
            nE0_mean_speed = self._get_mean_speed("-E0")
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            self._sum_E0_mean_speed += E0_mean_speed
            self._sum_nE0_mean_speed += nE0_mean_speed

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E26", "E25", "E24", "E23", "E22", "E21", "E20","-E1", "-E0", "E4", "E7", "E8", "E27", "-E27", "-E8", "-E7", "-E4", "E0"]
        refine_weights_roads = ["E26", "E25", "E24", "E23", "E22", "E21"]
        refine_weight = 10
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
                
                if road_id in refine_weights_roads:
                    self._waiting_times[car_id] = wait_time * refine_weight
                else:
                    self._waiting_times[car_id] = wait_time
                
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state

    def _set_phase(self, tlc_id, phase_code):
        #Activate the correct light combination in sumo
        traci.trafficlight.setPhase(tlc_id, phase_code)

    """
    def _set_yellow_phase(self, tlc_id, yellow_phase_code):
        #Activate the correct yellow light combination in sumo
        traci.trafficlight.setPhase(tlc_id, yellow_phase_code)

    def _set_red_phase(self, tlc_id, red_phase_code):
        #Activate the correct red light combination in sumo
        traci.trafficlight.setPhase(tlc_id, red_phase_code)

    def _set_green_phase(self, tlc_id, green_phase_code):
        #Activate the correct green light combination in sumo
        traci.trafficlight.setPhase(tlc_id, green_phase_code)
    """

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        #"-E1", "E0", "E22", "E21", "-E0"
        halt_E1 = traci.edge.getLastStepHaltingNumber("-E1")
        halt_E0 = traci.edge.getLastStepHaltingNumber("E0")
        halt_nE0 = traci.edge.getLastStepHaltingNumber("-E0")
        halt_E22 = traci.edge.getLastStepHaltingNumber("E22")
        halt_E21 = traci.edge.getLastStepHaltingNumber("E21")
        queue_length = halt_E1 + halt_E0 + halt_nE0 + halt_E22 + halt_E21
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        #incoming_roads = ["-E1", "E0", "E22", "E21", "-E0"]
        road1 = ["-E1"]
        road2 = ["E0"]
        road3 = ["E22", "E21"]
        road4 = ["-E0"]

        road1_car_num = [0, 0, 0, 0]

        #Count how many cars are on each road
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in road1:
                road1_car_num[0] = road1_car_num[0] + 1
            elif road_id in road2:
                road1_car_num[1] = road1_car_num[1] + 1
            elif road_id in road3:
                road1_car_num[2] = road1_car_num[2] + 1
            elif road_id in road4:
                road1_car_num[3] = road1_car_num[3] + 1

        for i in range(len(state)):
            road1_car_num[i] = float(road1_car_num[i] /30)
            if road1_car_num[i] >= 1:
                road1_car_num[i] = 1

            state[i] = road1_car_num[i]

        return state

    def _get_mean_speed(self, edge):
        """
        Retrieve the mean speed of every car in the incoming roads
        """
        mean_speed = traci.edge.getLastStepMeanSpeed(edge)

        return mean_speed
        

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._episode_E0_mean_speed_store.append(self._sum_E0_mean_speed / self._max_steps) # average mean speed of road E0 per step, in this episode
        self._episode_nE0_mean_speed_store.append(self._sum_nE0_mean_speed / self._max_steps) # average mean speed of road -E0 per step, in this episode

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

    @property
    def episode_E0_mean_speed_store(self):
        return self._episode_E0_mean_speed_store
    
    @property
    def episode_nE0_mean_speed_store(self):
        return self._episode_nE0_mean_speed_store