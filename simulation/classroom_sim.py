"""
Core SimPy simulation for smart classroom environment
"""

import simpy
import numpy as np
from simulation.ml_integration import predict_environment


DEFAULT_SIM_CONFIG = {
    'co2_production_per_student': 0.008,
    'co2_decay_rate': 0.02,
    'baseline_co2': 450,
    'heat_gain_per_student': 0.1,
    'light_heat_factor': 1 / 1000,
    'temperature_heat_scale': 0.01,
    'temperature_cooling_coeff': 0.05,
    'temperature_baseline': 22,
    'humidity_gain_per_student': 0.02,
    'humidity_vent_loss': 0.1,
    'light_day_base': 400,
    'light_day_amp': 200,
    'light_night_level': 100,
    'light_min': 100,
    'light_max': 1000,
    'occupancy_update_minutes': 1,
    'environment_update_minutes': 5,
    'monitor_interval_minutes': 10,
    'attention_persistence_cycles': 2,
    'attention_confidence_threshold': 0.7,
    'actuator_cooldown_minutes': 15,
}

ATTENTION_THRESHOLDS = {
    'temperature': {'low': 21.0, 'high': 25.0},
    'humidity': {'low': 40.0, 'high': 60.0},
    'co2': {'high': 800.0},
    'light': {'low': 500.0, 'high': 650.0},
}

COMFORT_THRESHOLDS = {
    'temperature': {'low': 20.0, 'high': 27.0},
    'humidity': {'low': 30.0, 'high': 60.0},
    'co2': {'high': 1000.0},
    'light': {'low': 300.0, 'high': 500.0},
}


def evaluate_features_zone(features):
    factor_zones = {}
    acceptable_factors = []
    non_conducive_factors = []

    for factor in ('temperature', 'humidity', 'co2', 'light'):
        value = float(features[factor])

        att = ATTENTION_THRESHOLDS[factor]
        comfort = COMFORT_THRESHOLDS[factor]

        att_low = att.get('low', -np.inf)
        att_high = att.get('high', np.inf)
        comfort_low = comfort.get('low', -np.inf)
        comfort_high = comfort.get('high', np.inf)

        if att_low <= value <= att_high:
            zone = 'optimal'
        elif comfort_low <= value <= comfort_high:
            zone = 'acceptable'
        else:
            zone = 'non-conducive'

        factor_zones[factor] = zone

        if zone == 'acceptable':
            acceptable_factors.append(f"{factor} outside attention band")

        if zone == 'non-conducive':
            non_conducive_factors.append(f"{factor} outside comfort band")

    if non_conducive_factors:
        overall_zone = 'non-conducive'
    elif acceptable_factors:
        overall_zone = 'acceptable'
    else:
        overall_zone = 'optimal'

    return {
        'overall_zone': overall_zone,
        'factor_zones': factor_zones,
        'acceptable_factors': acceptable_factors,
        'non_conducive_factors': non_conducive_factors,
    }


def fuse_model_zone_status(model_prediction, zone_state, confidence, low_confidence=0.6):
    overall_zone = zone_state.get('overall_zone', 'optimal')
    model_non_conducive = str(model_prediction) == 'non-conducive'

    if overall_zone == 'non-conducive':
        return {
            'final_status': 'non-conducive',
            'disagreement': not model_non_conducive,
            'rationale': 'Rule-based safety override',
        }

    if model_non_conducive and float(confidence) >= low_confidence:
        return {
            'final_status': 'non-conducive',
            'disagreement': True,
            'rationale': 'ML high-confidence risk',
        }

    if overall_zone == 'acceptable':
        return {
            'final_status': 'acceptable',
            'disagreement': model_non_conducive,
            'rationale': 'Comfort okay, attention not optimal',
        }

    return {
        'final_status': 'conducive',
        'disagreement': model_non_conducive,
        'rationale': 'All systems stable',
    }


def compute_agreement_score(model_prediction, zone_state):
    overall_zone = zone_state.get('overall_zone', 'optimal')
    model_conducive = str(model_prediction) == 'conducive'

    if overall_zone == 'non-conducive':
        return 1.0 if not model_conducive else 0.0

    if overall_zone == 'optimal':
        return 1.0 if model_conducive else 0.0

    return 0.5 if model_conducive else 0.0


class ClassroomSimulation:

    def __init__(self, env, num_students=30, room_size=100,
                 start_hour=9, initial_conditions=None,
                 sim_config=None, random_seed=None):

        self.env = env
        self.num_students = num_students
        self.room_size = room_size
        self.start_hour = start_hour

        self.sim_config = {**DEFAULT_SIM_CONFIG, **(sim_config or {})}
        if random_seed is not None:
            np.random.seed(int(random_seed))

        initial = initial_conditions or {}

        self.temperature = float(initial.get('temperature', self.sim_config['temperature_baseline']))
        self.co2 = float(initial.get('co2', self.sim_config['baseline_co2']))
        self.humidity = float(initial.get('humidity', 50))
        self.light = float(initial.get('light', 450))

        self.artificial_light_offset = 0.0
        self.log = []

        self.attention_drift_streak = 0
        self.acceptable_zone_streak = 0

        self.last_actuation_time = {
            'co2': -np.inf,
            'temperature': -np.inf,
            'humidity': -np.inf,
            'light': -np.inf,
        }

        self.total_actuations = 0

        self.env.process(self.simulate_occupancy())
        self.env.process(self.simulate_environment_changes())
        self.env.process(self.monitor_and_intervene())

    def simulate_occupancy(self):
        while True:
            co2_prod = self.num_students * self.sim_config['co2_production_per_student']
            decay = self.sim_config['co2_decay_rate'] * (self.co2 - self.sim_config['baseline_co2'])

            self.co2 += co2_prod - decay

            self.co2 = max(300, self.co2)

            yield self.env.timeout(self.sim_config['occupancy_update_minutes'])

    def simulate_environment_changes(self):
        while True:
            heat = self.num_students * self.sim_config['heat_gain_per_student']
            self.temperature += heat * self.sim_config['temperature_heat_scale']

            self.temperature -= self.sim_config['temperature_cooling_coeff'] * (
                self.temperature - self.sim_config['temperature_baseline']
            )

            self.humidity += (self.num_students * self.sim_config['humidity_gain_per_student']) - self.sim_config['humidity_vent_loss']
            self.humidity = np.clip(self.humidity, 0, 100)

            hour = (self.start_hour + self.env.now / 60) % 24

            if 6 <= hour <= 18:
                day_progress = (hour - 6.0) / 12.0
                # Smooth daytime curve: lower near 6/18, higher around noon.
                daylight = np.sin(np.pi * day_progress)
                natural_light = self.sim_config['light_day_base'] + self.sim_config['light_day_amp'] * daylight
            else:
                natural_light = self.sim_config['light_night_level']

            self.light = np.clip(
                natural_light + self.artificial_light_offset,
                self.sim_config['light_min'],
                self.sim_config['light_max']
            )

            yield self.env.timeout(self.sim_config['environment_update_minutes'])

    def evaluate_environment_zone(self, features):
        return evaluate_features_zone(features)

    def monitor_and_intervene(self):
        while True:

            features = {
                'temperature': self.temperature,
                'co2': self.co2,
                'humidity': self.humidity,
                'light': self.light,
            }

            try:
                prediction, confidence = predict_environment(features)
                confidence = float(confidence)
            except:
                prediction, confidence = "conducive", 0.8

            zone_state = self.evaluate_environment_zone(features)
            fused = fuse_model_zone_status(prediction, zone_state, confidence)

            final_status = fused['final_status']
            agreement_score = compute_agreement_score(prediction, zone_state)

            if zone_state['overall_zone'] == 'acceptable':
                self.acceptable_zone_streak += 1
            else:
                self.acceptable_zone_streak = 0

            interventions = []
            intervention_count = 0

            now = float(self.env.now)
            cooldown = float(self.sim_config['actuator_cooldown_minutes'])

            def can_actuate(factor):
                return (now - float(self.last_actuation_time[factor])) >= cooldown

            def record_actuation(factor, message):
                nonlocal intervention_count
                self.last_actuation_time[factor] = now
                self.total_actuations += 1
                intervention_count += 1
                interventions.append(message)

            # CO2 control
            if self.co2 > ATTENTION_THRESHOLDS['co2']['high'] and can_actuate('co2'):
                if self.co2 > COMFORT_THRESHOLDS['co2']['high']:
                    self.co2 = max(self.sim_config['baseline_co2'], self.co2 - 180)
                    record_actuation('co2', 'Emergency ventilation boost (CO2)')
                else:
                    self.co2 = max(self.sim_config['baseline_co2'], self.co2 - 90)
                    record_actuation('co2', 'Ventilation increased (CO2)')

            # Temperature control
            t_low = ATTENTION_THRESHOLDS['temperature']['low']
            t_high = ATTENTION_THRESHOLDS['temperature']['high']
            if self.temperature > t_high and can_actuate('temperature'):
                self.temperature = max(t_low, self.temperature - 0.8)
                record_actuation('temperature', 'Cooling pulse applied')
            elif self.temperature < t_low and can_actuate('temperature'):
                self.temperature = min(t_high, self.temperature + 0.8)
                record_actuation('temperature', 'Heating pulse applied')

            # Humidity control
            h_low = ATTENTION_THRESHOLDS['humidity']['low']
            h_high = ATTENTION_THRESHOLDS['humidity']['high']
            if self.humidity > h_high and can_actuate('humidity'):
                self.humidity = max(h_low, self.humidity - 3.0)
                record_actuation('humidity', 'Dehumidification cycle triggered')
            elif self.humidity < h_low and can_actuate('humidity'):
                self.humidity = min(h_high, self.humidity + 3.0)
                record_actuation('humidity', 'Humidification cycle triggered')

            # Lighting control via artificial lighting offset.
            l_low = ATTENTION_THRESHOLDS['light']['low']
            l_high = ATTENTION_THRESHOLDS['light']['high']
            if self.light < l_low and can_actuate('light'):
                self.artificial_light_offset += 60
                record_actuation('light', 'Lighting increased to attention range')
            elif self.light > l_high and can_actuate('light'):
                self.artificial_light_offset -= 60
                record_actuation('light', 'Lighting reduced to attention range')

            self.artificial_light_offset = float(np.clip(self.artificial_light_offset, -300, 400))

            zone_trigger_reason = fused['rationale']
            if zone_state['non_conducive_factors']:
                zone_trigger_reason = ' | '.join(zone_state['non_conducive_factors'])
            elif zone_state['acceptable_factors']:
                zone_trigger_reason = ' | '.join(zone_state['acceptable_factors'])

            interventions_triggered = '; '.join(interventions)

            model_zone_disagreement = int(bool(fused['disagreement']))

            self.log.append({
                'time': self.env.now,
                'temp': self.temperature,
                'co2': self.co2,
                'humidity': self.humidity,
                'light': self.light,
                'prediction': prediction,
                'model_prediction': prediction,
                'final_status': final_status,
                'agreement': agreement_score,
                'agreement_score': agreement_score,
                'zone': zone_state['overall_zone'],
                'overall_zone': zone_state['overall_zone'],
                'non_conducive_factors': '; '.join(zone_state['non_conducive_factors']),
                'acceptable_factors': '; '.join(zone_state['acceptable_factors']),
                'model_zone_disagreement': model_zone_disagreement,
                'interventions': interventions,
                'interventions_triggered': interventions_triggered,
                'intervention_count': intervention_count,
                'total_actuations': self.total_actuations,
                'zone_trigger_reason': zone_trigger_reason,
                'confidence': confidence,
            })

            yield self.env.timeout(self.sim_config['monitor_interval_minutes'])


def run_simulation(hours=2, num_students=30):
    env = simpy.Environment()
    sim = ClassroomSimulation(env, num_students=num_students)

    env.run(until=hours * 60)
    return sim.log