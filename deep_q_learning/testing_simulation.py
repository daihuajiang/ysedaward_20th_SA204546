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
=====以下action先不模擬=======
2: 台2線和台2乙線紅綠燈階段0
3: 台2線和台2乙線紅綠燈階段1
4: action 0+2
5: action 0+3
6: action 1+2
7: action 1+3
"""


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, red_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._red_duration = red_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_E0_mean_speed = 0
        self._sum_nE0_mean_speed = 0
        old_total_wait = 0
        old_action = -1 # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

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
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E26", "E25", "E24", "E23", "E22", "E21", "E20","-E1", "-E0", "E4", "E7", "E8", "E27", "-E27", "-E8", "-E7", "-E4", "E0"]
        #refine_weights_roads = ["E26", "E25", "E24", "E23", "E22", "E21"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
                """
                if road_id in refine_weights_roads:
                    self._waiting_times[car_id] = wait_time * 3.5
                else:
                    self._waiting_times[car_id] = wait_time
                """
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))

    def _set_phase(self, tlc_id, phase_code):
        #Activate the correct light combination in sumo
        traci.trafficlight.setPhase(tlc_id, phase_code)

    """
    def _set_yellow_phase(self, old_action):
        
        #Activate the correct yellow light combination in sumo
        
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        
        #Activate the correct green light combination in sumo

        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
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
        #incoming_roads = ["-E1", "E0", "E22", "E20", "E2", "E14", "-E13"]
        road1 = ["-E1"]
        road2 = ["E0"]
        road3 = ["E22", "J17", "E21"]
        road4 = ["-E0"]

        road1_car_num = [0, 0, 0, 0]

        #計算每條路上有多少台車
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

    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode

