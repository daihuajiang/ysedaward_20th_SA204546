import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated*2)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            #三芝-台2乙線-竹圍: id="R1" edges="E14 -E11 -E2 -E28 -E27 -E8 -E7 -E4 -E0 E1"
            #淡水-台2乙線-竹圍: id="R2" edges="-E13 -E11 -E2 -E28 -E27 -E8 -E7 -E4 -E0 E1"
            #三芝-中正東路-竹圍: id="R3" edges="E14 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 E1"
            #淡水-中正東路-竹圍: id="R4" edges="-E13 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 E1"
            #竹圍-三芝: id="R5" edges="-E1 E0 E4 E7 E8 E27 E28 E2 E12 E9"
            #竹圍-淡水: id="R6" edges="-E1 E0 E4 E7 E8 E27 E28 E2 E11 E13"
            #三芝-八勢路: id="R7" edges="E14 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 -E20"
            #淡水-八勢路: id="R8" edges="-E13 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 -E20"
            #八勢路-三芝: id="R9" edges="E20 E0 E4 E7 E8 E27 E28 E2 E12 E9"
            #八勢路-淡水: id="R10" edges="E20 E0 E4 E7 E8 E27 E28 E2 E11 E13"
            #三芝-中正東路-淡水: id="R11" edges="E14 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 E0 E4 E7 E8 E27 E28 E2 E11 E13"
            #淡水-中正東路-三芝: id="R12" edges="-E13 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 E0 E4 E7 E8 E27 E28 E2 E12 E9"


            print("""<routes>
            <vType id="PassengerCar" minGap="1.0" maxSpeed="50.00" vClass="passenger" color="yellow" accel="3.0" decel="4.5" sigma="0.5"/>
            <route id="R1" edges="E14 -E11 -E2 -E28 -E27 -E8 -E7 -E4 E0 E1"/>
            <route id="R2" edges="-E13 -E11 -E2 -E28 -E27 -E8 -E7 -E4 E0 E1"/>
            <route id="R3" edges="E14 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 E1"/>
            <route id="R4" edges="-E13 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 E1"/>
            <route id="R5" edges="-E1 -E0 E4 E7 E8 E27 E28 E2 E12 E9"/>
            <route id="R6" edges="-E1 -E0 E4 E7 E8 E27 E28 E2 E11 E13"/>
            <route id="R7" edges="E14 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 -E20"/>
            <route id="R8" edges="-E13 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 -E20"/>
            <route id="R9" edges="E20 -E0 E4 E7 E8 E27 E28 E2 E12 E9"/>
            <route id="R10" edges="E20 -E0 E4 E7 E8 E27 E28 E2 E11 E13"/>
            <route id="R11" edges="E14 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 -E0 E4 E7 E8 E27 E28 E2 E11 E13"/>
            <route id="R12" edges="-E13 -E11 -E2 -E28 -E27 E26 E25 E24 E23 E22 E21 -E0 E4 E7 E8 E27 E28 E2 E12 E9"/>""", file=routes)

            #隨機生成車流
            for car_counter, step in enumerate(car_gen_steps):
                route_random = np.random.uniform()
                if route_random < 0.0225:  #三芝-中正東路-淡水
                    print('    <vehicle id="R11_%i" type="PassengerCar" route="R11" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.0225 and route_random < 0.0250:  #三芝-八勢路
                    print('    <vehicle id="R7_%i" type="PassengerCar" route="R7" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.0250 and route_random < 0.0475:  #淡水-中正東路-三芝
                    print('    <vehicle id="R12_%i" type="PassengerCar" route="R12" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.0475 and route_random < 0.05:  #淡水-八勢路
                    print('    <vehicle id="R8_%i" type="PassengerCar" route="R8" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.05 and route_random < 0.075:  #三芝-中正東路-竹圍
                    print('    <vehicle id="R3_%i" type="PassengerCar" route="R3" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.075 and route_random < 0.1:  #淡水-中正東路-竹圍
                    print('    <vehicle id="R4_%i" type="PassengerCar" route="R4" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.1 and route_random < 0.3:  #三芝-台2乙線-竹圍
                    print('    <vehicle id="R1_%i" type="PassengerCar" route="R1" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.3 and route_random < 0.5:  #淡水-台2乙線-竹圍
                    print('    <vehicle id="R2_%i" type="PassengerCar" route="R2" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.5 and route_random < 0.515:  #八勢路-三芝
                    print('    <vehicle id="R9_%i" type="PassengerCar" route="R9" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.515 and route_random < 0.53:  #八勢路-淡水
                    print('    <vehicle id="R10_%i" type="PassengerCar" route="R10" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                elif route_random >= 0.53 and route_random < 0.765:  #竹圍-三芝
                    print('    <vehicle id="R5_%i" type="PassengerCar" route="R5" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                else:  #竹圍-淡水
                    print('    <vehicle id="R6_%i" type="PassengerCar" route="R6" depart="%s" departLane="random" />' % (car_counter, step), file=routes)
                   
            print("</routes>", file=routes)
