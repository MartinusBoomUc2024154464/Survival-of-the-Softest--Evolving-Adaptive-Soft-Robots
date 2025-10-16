import pandas as pd
from scipy.stats import kruskal, f_oneway, shapiro
import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import pandas as pd
import matplotlib.pyplot as plt
from neural_controller import *
import os
import glob
import json


DEBUG_MODE = False  

# ---- PARAMETERS ----
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid 
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
SCENARIO = 'Walker-v0'
CONTROLLER = alternating_gait
SEEDS = [42] if DEBUG_MODE else [42, 123, 999, 2025, 7]

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

# ---- GA and RS PARAMETERS ---- 
CROSSOVER_TYPE = "one_point"  # "one_point", "two-point" ou "uniform"
CROSSOVER_RATE = 0.8
MUTATION_TYPE = "flip"      # "flip", "swap" ou "scramble"
MUTATION_RATE = 0.2
SELECTION_TYPE = "tournament"  # "tournament" ou "roulette"
TOURNAMENT_SIZE = 5
ELISTIM = 0.1
POP_SIZE = 15 if DEBUG_MODE else 30
GENERATIONS = 20 if DEBUG_MODE else 50
STEPS = 500 
NUM_GENERATIONS = POP_SIZE * (1 + GENERATIONS)
ESTAGNATION_WINDOW = 10
IMPROVEMENT_THRESHOLD = 1e-3

# FITNESS FUNCTION
def evaluate_fitness(robot_structure, view=False):    
    try:
        try:
            connectivity = get_full_connectivity(robot_structure)
        except Exception:
            return -3.0 
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')
        for t in range(STEPS):  
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0

# RANDOM ROBOT FUNCTION
def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

# RANDOM SEARCH ALGORITHM
def random_search():
    """Perform a random search to find the best robot structure."""
    best_robot = None
    best_fitness = -float('inf')
    all_fitnesses = []

    for it in range(NUM_GENERATIONS):
        robot = create_random_robot() 
        fitness_score = evaluate_fitness(robot)
        all_fitnesses.append(fitness_score)

        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot
        
        print(f"Iteration {it + 1}: Fitness = {fitness_score}")
    
    return best_robot, best_fitness, all_fitnesses

# MUTATION FUNCTIONS
def mutation_swap(structure: np.ndarray) -> np.ndarray:
    structure = copy.deepcopy(structure)
    flat = structure.flatten()
    i, j = np.random.choice(len(flat), size=2, replace=False)
    flat[i], flat[j] = flat[j], flat[i]
    return flat.reshape(structure.shape)

def mutation_flip(structure: np.ndarray) -> np.ndarray:
    structure = copy.deepcopy(structure)
    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            current = structure[i, j]
            structure[i, j] = 0 if current > 0 else np.random.choice([1, 2, 3, 4])
    return structure

def mutation_scramble(structure: np.ndarray) -> np.ndarray:
    structure = copy.deepcopy(structure)
    flat = structure.flatten()
    start, end = sorted(np.random.choice(len(flat), size=2, replace=False))
    scrambled = flat[start:end].copy()
    np.random.shuffle(scrambled)
    flat[start:end] = scrambled
    return flat.reshape(structure.shape)

def mutate_ga(structure: np.ndarray) -> np.ndarray:
    if MUTATION_TYPE == "flip":
        return mutation_flip(structure)
    elif MUTATION_TYPE == "swap":
        return mutation_swap(structure)
    elif MUTATION_TYPE == "scramble":
        return mutation_scramble(structure)
    else:
        return mutation_flip(structure)

# CROSSOVER FUNCTIONS
def crossover_one_point(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)
    x_split = np.random.randint(1, parent1.shape[0])
    y_split = np.random.randint(1, parent1.shape[1])
    parent1[:x_split, :y_split] = parent2[:x_split, :y_split]
    return parent1

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)
    mask = np.random.rand(*parent1.shape) < 0.5
    return np.where(mask, parent1, parent2)

def crossover_two_point(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    a, b = sorted(np.random.choice(len(flat1), 2, replace=False))
    flat1[a:b] = flat2[a:b]
    return flat1.reshape(parent1.shape)

def crossover_ga(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    if CROSSOVER_TYPE == "uniform":
        return crossover_uniform(parent1, parent2)
    elif CROSSOVER_TYPE == "one_point":
        return crossover_one_point(parent1, parent2)
    elif CROSSOVER_TYPE == "two_point":
        return crossover_two_point(parent1, parent2)
    else:
        return crossover_one_point(parent1, parent2)

# SELECTION FUNCTIONS
def tournament_selection(population: list[np.ndarray], fitness_scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    selected1 = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
    best_index1 = selected1[np.argmax([fitness_scores[i] for i in selected1])]
    
    selected2 = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
    best_index2 = selected2[np.argmax([fitness_scores[i] for i in selected2])]
    
    return population[best_index1], population[best_index2]

def roulette_wheel_selection(population: list[np.ndarray], fitness_scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    min_fitness = min(fitness_scores)
    if min_fitness < 0:
        norm_fitness = [f - min_fitness for f in fitness_scores]
    else:
        norm_fitness = list(fitness_scores)

    total_fitness = sum(norm_fitness)
    if total_fitness == 0:
        probs = [1 / len(norm_fitness)] * len(norm_fitness)
    else:
        probs = [f / total_fitness for f in norm_fitness]

    selected_indices = np.random.choice(len(population), 2, p=probs)
    return population[selected_indices[0]], population[selected_indices[1]]

def selection_ga(population: list[np.ndarray], fitness_scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if SELECTION_TYPE == "roulette":
        return roulette_wheel_selection(population, fitness_scores)
    else:
        return tournament_selection(population, fitness_scores)
    
# GENETIC ALGORITHM
def genetic_algorithm():
    # Initialize the population with valid connected robot structures
    population = []
    disconnected_count = 0
    while len(population) < POP_SIZE:
        robot = create_random_robot()
        if np.any(robot > 0) and is_connected(robot):
            population.append(robot)
        else:
            disconnected_count += 1
    print(f"Filtered out {disconnected_count} disconnected structures during initialization.")

    # Evaluate initial fitness and set best individual
    fitness_scores = [evaluate_fitness(robot) for robot in population]
    best_structure = population[np.argmax(fitness_scores)]
    best_fitness = max(fitness_scores)

    best_fitnesses = []
    avg_fitnesses = []
    all_fitnesses = [fitness_scores.copy()]

    for gen in range(GENERATIONS):
        pop_array = np.array([ind.flatten() for ind in population])
        n_unique = np.unique(pop_array, axis=0).shape[0]
        print(f"GA Gen {gen+1}/{GENERATIONS} | Best: {best_fitness:.2f} | Avg: {np.mean(fitness_scores):.2f} | Unique: {n_unique}/{len(population)}")

        best_fitnesses.append(np.max(fitness_scores))
        avg_fitnesses.append(np.mean(fitness_scores))

        # Select elites (top individuals)
        elite_size = max(1, int(POP_SIZE * ELISTIM))
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
        elites = [copy.deepcopy(population[i]) for i in elite_indices]

        # Generate offspring
        filtered_offspring = 0
        offspring = []
        while len(offspring) < POP_SIZE - elite_size:
            parent1, parent2 = selection_ga(population, fitness_scores)
            child = copy.deepcopy(parent1)
            if np.random.rand() < CROSSOVER_RATE:
                child = crossover_ga(child, parent2)
            if np.random.rand() < MUTATION_RATE:
                child = mutate_ga(child)
            if np.any(child > 0) and is_connected(child):
                offspring.append(child)
            else:
                filtered_offspring += 1

        # Remove duplicates
        offspring_flat = np.array([ind.flatten() for ind in offspring])
        _, unique_idx = np.unique(offspring_flat, axis=0, return_index=True)
        unique_offspring = [offspring[i] for i in sorted(unique_idx)]
        n_eliminated = len(offspring) - len(unique_offspring)
        print(f"Filtered out {filtered_offspring} disconnected offspring and {n_eliminated} clones in generation {gen+1}.")

        # Fill with new valid random individuals
        while len(unique_offspring) < POP_SIZE - elite_size:
            robot = create_random_robot()
            if np.any(robot > 0) and is_connected(robot):
                unique_offspring.append(robot)

        # Form new population
        population = elites + unique_offspring[:POP_SIZE - elite_size]
        pop_array = np.array([ind.flatten() for ind in population])
        n_unique = np.unique(pop_array, axis=0).shape[0]
        print(f"After filtering, unique individuals: {n_unique}/{len(population)}")

        fitness_scores = [evaluate_fitness(robot) for robot in population]
        all_fitnesses.append(fitness_scores.copy())

        # Update best structure
        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_structure = population[np.argmax(fitness_scores)]

    return best_structure, best_fitness, best_fitnesses, avg_fitnesses, all_fitnesses

# PHASE 1 FUNCTIONS 
def run_stagnation_analysis(csv_path, controller_set, scenario_set):
    df_phase1 = pd.read_csv(csv_path)
    stagnation_points = []

    for scenario in scenario_set:
        plt.figure(figsize=(12, 6))
        max_stag_scenario = 0

        for _, ctrl_name in controller_set:
            df_filtered = df_phase1[
                (df_phase1['scenario'] == scenario) &
                (df_phase1['controller'] == ctrl_name)
            ]

            print(f"\nStagnation Detection: {ctrl_name} ({scenario})")

            bests_by_seed = []
            for seed in df_filtered['seed'].unique():
                seed_group = df_filtered[df_filtered['seed'] == seed]
                bests = seed_group.groupby('generation')['fitness'].max().reset_index().sort_values('generation')

                bests_by_seed.append(bests.set_index('generation')['fitness'])

                no_improvement_count = 0
                stagnation_gen = None
                last_best = bests['fitness'].iloc[0]

                for i in range(1, len(bests)):
                    current = bests['fitness'].iloc[i]
                    if abs(current - last_best) < IMPROVEMENT_THRESHOLD:
                        no_improvement_count += 1
                        if no_improvement_count >= ESTAGNATION_WINDOW:
                            stagnation_gen = bests['generation'].iloc[i - ESTAGNATION_WINDOW + 1]
                            break
                    else:
                        no_improvement_count = 0
                        last_best = current

                if stagnation_gen is not None:
                    max_stag_scenario = max(max_stag_scenario, stagnation_gen)
                else:
                    max_stag_scenario = max(max_stag_scenario, GENERATIONS)

            mean_curve = pd.concat(bests_by_seed, axis=1).mean(axis=1)

            plt.plot(mean_curve.index, mean_curve.values, label=f"{ctrl_name} Mean", linewidth=2)

            plt.axvline(x=max_stag_scenario, linestyle='--', color='gray', alpha=0.6)

            print(f"Max stagnation generation for {ctrl_name} on {scenario}: {max_stag_scenario}")
            stagnation_points.append({
                'controller': ctrl_name,
                'scenario': scenario,
                'max_stagnation_gen': max_stag_scenario
            })

        plt.title(f"Mean Fitness Evolution - Phase 1 ({scenario})")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.legend(title="Controller", loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        output_dir = f"plots/fase1/phase1/{scenario}/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}phase1_mean_fitness_per_controller_{scenario}.png")
        plt.show()

    df_stagnation = pd.DataFrame(stagnation_points)
    os.makedirs("csvs/fase1", exist_ok=True)
    df_stagnation.to_csv("csvs/fase1/phase1_stagnation.csv", index=False)

def run_controller_comparison_boxplot(csv_path, controller_set, scenario_set):
    df_phase1 = pd.read_csv(csv_path)

    for scenario in scenario_set:
        plt.figure()
        data = [df_phase1[
            (df_phase1['scenario'] == scenario) &
            (df_phase1['controller'] == ctrl_name)
        ].groupby('seed')['fitness'].max().values for _, ctrl_name in controller_set]
        labels = [ctrl_name for _, ctrl_name in controller_set]

        box = plt.boxplot(data, patch_artist=True, tick_labels=labels)

        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        means = [np.mean(values) for values in data]
        for i, mean in enumerate(means):
            plt.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=9)

        plt.title(f"Controller Comparison using GA - {scenario}")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("plots/fase1/phase1/boxplots", exist_ok=True)
        plt.savefig(f"plots/fase1/phase1/boxplots/controller_comparison_ga_{scenario}.png")
        plt.show()

def run_phase1(controller_set, scenario_set):
    all_results_phase1 = {}
    csv_data_phase1 = []
    best_controllers = {}

    for scenario in scenario_set:
        best_controller = None
        best_avg_fitness = -float('inf')

        for ctrl_fn, ctrl_name in controller_set:
            temp_fitnesses = []
            print(f"Evaluating {ctrl_name} on {scenario}")
            for seed in SEEDS:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                global CONTROLLER, SCENARIO
                CONTROLLER = ctrl_fn
                SCENARIO = scenario
                
                best_robot, best_fitness, best_fitnesses, avg_fitnesses, all_fitnesses = genetic_algorithm()

                temp_fitnesses.append(best_fitness)

                for gen_idx, fitness_list in enumerate(all_fitnesses):
                    for ind_idx, fit in enumerate(fitness_list):
                        csv_data_phase1.append({
                            'seed': seed,
                            'generation': gen_idx,
                            'individual': ind_idx,
                            'fitness': fit,
                            'controller': ctrl_name,
                            'scenario': scenario
                        })

            avg_fitness = np.mean(temp_fitnesses)
            all_results_phase1[f"{ctrl_name}_{scenario}"] = temp_fitnesses
            if avg_fitness > best_avg_fitness:
                best_avg_fitness = avg_fitness
                best_controller = (ctrl_fn, ctrl_name)

        best_controllers[scenario] = best_controller

        df_best_controllers = pd.DataFrame([
            {'scenario': scenario, 'controller': best_controllers[scenario][1]}
            for scenario in scenario_set
        ])
        os.makedirs("csvs/fase1", exist_ok=True)
        df_best_controllers.to_csv("csvs/fase1/phase1_best_controllers.csv", index=False)

    df_phase1 = pd.DataFrame(csv_data_phase1)
    os.makedirs("csvs/fase1", exist_ok=True)
    df_phase1.to_csv("csvs/fase1/phase1.csv", index=False)

    for scenario in scenario_set:
        ctrl_fn, ctrl_name = best_controllers[scenario]
        print(f"Best Controller for {scenario} is {ctrl_name}")

    return all_results_phase1, best_controllers, scenario_set, controller_set

# PHASE 2 FUNCTIONS
def format_label(alg, scenario):
    name = alg.replace("GA_", "").replace(f"_{scenario}", "")
    parts = name.split("_")
    if len(parts) == 4:
        return f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[3]}"
    elif len(parts) == 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return name

def plot_phase2_comparison_boxplots(csv_path: str, scenario_set: list[str]):
    df = pd.read_csv(csv_path)

    for scenario in scenario_set:
        plt.figure(figsize=(18, 7))
        df_scenario = df[df['scenario'] == scenario]
        grouped = df_scenario.groupby(['algorithm', 'seed'])['fitness'].max().reset_index()
        algorithms = grouped['algorithm'].unique()

        # Use format_label for improved label readability
        labels = [format_label(alg, scenario) for alg in algorithms]

        data = [grouped[grouped['algorithm'] == alg]['fitness'].values for alg in algorithms]

        box = plt.boxplot(data, patch_artist=True, labels=labels)
        colors = plt.cm.Set3.colors
        for patch, color in zip(box['boxes'], colors * (len(data) // len(colors) + 1)):
            patch.set_facecolor(color)

        means = [np.mean(values) for values in data]
        for i, mean in enumerate(means):
            plt.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=8)

        plt.title(f"Algorithm Comparison - {scenario}", fontsize=14, fontweight='bold', loc='center')
        plt.ylabel("Fitness")
        plt.xticks(rotation=35, ha='right')
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        os.makedirs("plots/fase1/phase2/boxplots", exist_ok=True)
        plt.savefig(f"plots/fase1/phase2/boxplots/phase2_comparison_boxplot_{scenario}.png")
        plt.show()

def plot_phase2_average_curves(csv_path: str, scenario_set: list[str]):
    df = pd.read_csv(csv_path)

    for scenario in scenario_set:
        df_scenario = df[df['scenario'] == scenario]
        df_scenario = df_scenario[df_scenario['algorithm'].str.startswith("GA_")]

        unique_algorithms = df_scenario['algorithm'].unique()

        plt.figure(figsize=(18, 8))
        for algorithm in unique_algorithms:
            label = format_label(algorithm, scenario)
            df_alg = df_scenario[df_scenario['algorithm'] == algorithm]
            gen_fitness = df_alg[df_alg['generation'] != 'best']
            gen_fitness['generation'] = gen_fitness['generation'].astype(int)
            mean_curve = gen_fitness.groupby('generation')['fitness'].mean()
            plt.plot(mean_curve.index, mean_curve.values, label=label, linewidth=2.0)

        plt.title(f"Mean Fitness Evolution for GA Combinations - {scenario}", fontsize=14, fontweight='bold', loc='center')
        plt.xlabel("Generation")
        plt.ylabel("Mean Fitness")
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        os.makedirs(f"plots/fase1/phase2/average_curves", exist_ok=True)
        plt.savefig(f"plots/fase1/phase2/average_curves/mean_curve_ga_{scenario}.png")
        plt.show()

def run_phase2_scenario(scenario, best_controllers, max_gen_per_scenario):
    ctrl_fn, ctrl_name = best_controllers[scenario]
    max_gen = max_gen_per_scenario[scenario]

    global GENERATIONS, NUM_GENERATIONS, CONTROLLER, SCENARIO
    GENERATIONS = max_gen
    NUM_GENERATIONS = POP_SIZE * (1 + max_gen)

    print(f"\nPhase 2: Best Controller for {scenario} is {ctrl_name} | Generations on GA: {GENERATIONS} | Evaluations: {NUM_GENERATIONS}")

    crossover_options = ["one_point", "two_point", "uniform"]
    mutation_options = ["flip", "swap", "scramble"]
    selection_options = ["tournament", "roulette"]

    all_results_phase2 = {}
    csv_data_phase2 = []

    for crossover_type in crossover_options:
        for mutation_type in mutation_options:
            for selection_type in selection_options:
                print(f"\nTesting GA with Crossover: {crossover_type}, Mutation: {mutation_type}, Selection: {selection_type}")
                global CROSSOVER_TYPE, MUTATION_TYPE, SELECTION_TYPE
                CROSSOVER_TYPE = crossover_type
                MUTATION_TYPE = mutation_type
                SELECTION_TYPE = selection_type

                config_label = f"GA_{crossover_type}_{mutation_type}_{selection_type}_{scenario}"
                fitnesses_config = []

                for seed in SEEDS:
                    print(f"\nTesting {config_label} on {scenario}, Seed: {seed}")
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    CONTROLLER = ctrl_fn
                    SCENARIO = scenario

                    best_robot, best_fitness, best_fitnesses, avg_fitnesses, all_fitnesses = genetic_algorithm()
                    fitnesses_config.append(best_fitness)

                    for gen_idx, fitness_list in enumerate(all_fitnesses):
                        for ind_idx, fit in enumerate(fitness_list):
                            csv_data_phase2.append({
                                'seed': seed,
                                'generation': gen_idx,
                                'individual': ind_idx,
                                'fitness': fit,
                                'controller': ctrl_name,
                                'scenario': scenario,
                                'algorithm': config_label
                            })
                    csv_data_phase2.append({
                        'seed': seed,
                        'generation': 'best',
                        'individual': 'best',
                        'fitness': best_fitness,
                        'controller': ctrl_name,
                        'scenario': scenario,
                        'algorithm': config_label,
                        'best_robot': json.dumps(best_robot.tolist())
                    })

                all_results_phase2[config_label] = fitnesses_config
                os.makedirs(f"gifs/fase1/phase2/{scenario}", exist_ok=True)
                utils.create_gif(best_robot, filename=f'gifs/fase1/phase2/{scenario}/genetic_algorithm_{config_label}.gif', scenario=scenario, steps=STEPS, controller=ctrl_fn)

    # RS (Random Search)
    rs_fitnesses = []
    for seed in SEEDS:
        print(f"\nTesting RS with {ctrl_name} on {scenario}, Seed: {seed}")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        CONTROLLER = ctrl_fn
        SCENARIO = scenario

        best_robot, best_fitness, all_fitnesses = random_search()
        rs_fitnesses.append(best_fitness)

        for gen_idx, fitness in enumerate(all_fitnesses):
            csv_data_phase2.append({
                'seed': seed,
                'generation': gen_idx,
                'individual': 0,
                'fitness': fitness,
                'controller': ctrl_name,
                'scenario': scenario,
                'algorithm': 'RS'
            })
        csv_data_phase2.append({
            'seed': seed,
            'generation': 'best',
            'individual': 'best',
            'fitness': best_fitness,
            'controller': ctrl_name,
            'scenario': scenario,
            'algorithm': 'RS',
            'best_robot': json.dumps(best_robot.tolist())
        })

    all_results_phase2[f"RS_{scenario}"] = rs_fitnesses
    os.makedirs(f"gifs/fase1/phase2/{scenario}", exist_ok=True)
    utils.create_gif(best_robot, filename=f'gifs/fase1/phase2/{scenario}/random_search_{scenario}.gif', scenario=scenario, steps=STEPS, controller=ctrl_fn)

    df_phase2 = pd.DataFrame(csv_data_phase2)
    output_dir = f"csvs/fase1/phase2"
    os.makedirs(output_dir, exist_ok=True)
    df_phase2.to_csv(f"{output_dir}/phase2_{scenario}.csv", index=False)

    return all_results_phase2

def run_phase2(scenario_set, best_controllers_path):
    df_stagnation = pd.read_csv("csvs/fase1/phase1_stagnation.csv")

    df_best_controllers = pd.read_csv(best_controllers_path)
    controller_name_to_function = {
        "Alternating": alternating_gait,
        "Sinusoidal": sinusoidal_wave,
        "Hopping": hopping_motion
    }
    best_controllers = {
        row['scenario']: (controller_name_to_function[row['controller']], row['controller'])
        for _, row in df_best_controllers.iterrows()
    }

    max_gen_per_scenario = {}
    for scenario in scenario_set:
        ctrl_name = best_controllers[scenario][1]
        filtered = df_stagnation[
            (df_stagnation['scenario'] == scenario) & 
            (df_stagnation['controller'] == ctrl_name)
        ]
        max_gen = filtered['max_stagnation_gen'].max()
        max_gen_per_scenario[scenario] = int(max_gen)

    all_results_phase2 = {}
    global_csv_data = []

    for scenario in scenario_set:
        scenario_results = run_phase2_scenario(scenario, best_controllers, max_gen_per_scenario)
        all_results_phase2.update(scenario_results)
        scenario_csv_path = f"csvs/fase1/phase2/phase2_{scenario}.csv"
        if os.path.exists(scenario_csv_path):
            df_scenario = pd.read_csv(scenario_csv_path)
            global_csv_data.append(df_scenario)

    if global_csv_data:
        df_phase2 = pd.concat(global_csv_data, ignore_index=True)
        os.makedirs("csvs/fase1/phase2", exist_ok=True)
        df_phase2.to_csv("csvs/fase1/phase2/phase2.csv", index=False)

    return all_results_phase2

# PHASE 3 FUNCTIONS
def run_phase3(csv_path):
    """
    Perform statistical comparison of GA algorithm results from phase2.csv.
    For each scenario:
      - Tests normality.
      - Applies ANOVA or Kruskal-Wallis.
      - Identifies the best GA by mean fitness.
      - Saves statistical results and best GA into separate CSVs per scenario.
    """
    os.makedirs("csvs/fase1/phase3", exist_ok=True)

    df = pd.read_csv(csv_path)
    scenario_set = df['scenario'].unique()

    for scenario in scenario_set:
        df_scenario = df[df['scenario'] == scenario]
        df_scenario = df_scenario[df_scenario['algorithm'].str.startswith('GA_')]

        if df_scenario.empty:
            print(f"No GA data for scenario {scenario}. Skipping.")
            continue

        grouped = df_scenario[df_scenario['generation'] != 'best'].groupby(['algorithm', 'seed'])['fitness'].max().reset_index()

        algorithms = grouped['algorithm'].unique()
        data_per_algorithm = [grouped[grouped['algorithm'] == alg]['fitness'].values for alg in algorithms]

        normal = True
        for data in data_per_algorithm:
            if len(data) < 3:
                normal = False
                break
            stat, p = shapiro(data)
            if p < 0.05:
                normal = False
                break

        if normal:
            test_used = "ANOVA"
            stat, p_value = f_oneway(*data_per_algorithm)
        else:
            test_used = "Kruskal-Wallis"
            stat, p_value = kruskal(*data_per_algorithm)

        print(f"\nScenario: {scenario}")
        print(f"Test Used: {test_used}")
        print(f"Statistic={stat:.4f}, p-value={p_value:.4f}")

        if p_value < 0.05:
            print("There are statistically significant differences between algorithms.")
        else:
            print("There are no statistically significant differences between algorithms.")

        significance = "Yes" if p_value < 0.05 else "No"

        result_data = {
            'Scenario': [scenario],
            'Test Used': [test_used],
            'Statistic': [stat],
            'p-value': [p_value],
            'Significant': [significance]
        }
        df_result = pd.DataFrame(result_data)
        df_result.to_csv(f"csvs/fase1/phase3/statistical_test_{scenario}.csv", index=False)

        mean_fitness_per_algorithm = grouped.groupby('algorithm')['fitness'].mean()
        best_algorithm = mean_fitness_per_algorithm.idxmax()

        best_split = best_algorithm.split("_")
        best_crossover = best_split[1]
        best_mutation = best_split[2]
        best_selection = best_split[3]

        best_ga_data = {
            'scenario': [scenario],
            'best_crossover': [best_crossover],
            'best_mutation': [best_mutation],
            'best_selection': [best_selection]
        }
        df_best_ga = pd.DataFrame(best_ga_data)
        df_best_ga.to_csv(f"csvs/fase1/phase3/best_ga_{scenario}.csv", index=False)

def plot_phase3_summary_tables():
    """ Summarize and display phase 3 results: statistical tests and best GA combinations. """
    os.makedirs("plots/fase1/phase3", exist_ok=True)

    test_files = glob.glob("csvs/fase1/phase3/statistical_test_*.csv")
    best_ga_files = glob.glob("csvs/fase1/phase3/best_ga_*.csv")

    df_tests = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
    df_best_gas = pd.concat([pd.read_csv(f) for f in best_ga_files], ignore_index=True)

    df_summary = df_tests.merge(df_best_gas, left_on="Scenario", right_on="scenario")

    df_summary.to_csv("csvs/fase1/phase3/summary_phase3.csv", index=False)
    print("\nPhase 3 Statistical Summary")
    print(df_summary)

# PHASE 4 FUNCTIONS
def run_phase4(scenario_set):
    """
    Perform hyperparameter tuning for the best GA combination from phase 3.
    Varies crossover rate, mutation rate, and elitism rate.
    Saves all results into phase4.csv.
    """
    crossover_rates = [0.6, 0.8, 1.0]
    mutation_rates = [0.1, 0.2, 0.3]
    elitism_rates = [0.1]

    os.makedirs("csvs/fase1/phase4", exist_ok=True)

    global_csv_data = []

    df_best_controllers = pd.read_csv("csvs/fase1/phase1_best_controllers.csv")
    controller_name_to_function = {
        "Alternating": alternating_gait,
        "Sinusoidal": sinusoidal_wave,
        "Hopping": hopping_motion
    }
    best_controllers = {
        row['scenario']: (controller_name_to_function[row['controller']], row['controller'])
        for _, row in df_best_controllers.iterrows()
    }

    for scenario in scenario_set:
        best_ga_path = f"csvs/fase1/phase3/best_ga_{scenario}.csv"
        if not os.path.exists(best_ga_path):
            print(f"Best GA config for {scenario} not found. Skipping.")
            continue

        df_best = pd.read_csv(best_ga_path)
        best_crossover = df_best.loc[0, 'best_crossover']
        best_mutation = df_best.loc[0, 'best_mutation']
        best_selection = df_best.loc[0, 'best_selection']

        ctrl_fn, ctrl_name = best_controllers[scenario]

        print(f"\nPhase 4 Hyperparameter Tuning for {scenario}: {best_crossover}, {best_mutation}, {best_selection}")

        for crossover_rate in crossover_rates:
            for mutation_rate in mutation_rates:
                for elitism_rate in elitism_rates:
                    label = f"GA_Hyper_{crossover_rate}_{mutation_rate}_{elitism_rate}_{scenario}"

                    global CROSSOVER_TYPE, MUTATION_TYPE, SELECTION_TYPE
                    global CROSSOVER_RATE, MUTATION_RATE, ELISTIM
                    CROSSOVER_TYPE = best_crossover
                    MUTATION_TYPE = best_mutation
                    SELECTION_TYPE = best_selection
                    CROSSOVER_RATE = crossover_rate
                    MUTATION_RATE = mutation_rate
                    ELISTIM = elitism_rate

                    for seed in SEEDS:
                        print(f"\nTesting {label} on {scenario}, Seed: {seed}")
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.manual_seed(seed)
                        global CONTROLLER, SCENARIO
                        CONTROLLER = ctrl_fn
                        SCENARIO = scenario

                        best_robot, best_fitness, best_fitnesses, avg_fitnesses, all_fitnesses = genetic_algorithm()

                        for gen_idx, fitness_list in enumerate(all_fitnesses):
                            for ind_idx, fit in enumerate(fitness_list):
                                global_csv_data.append({
                                    'seed': seed,
                                    'generation': gen_idx,
                                    'individual': ind_idx,
                                    'fitness': fit,
                                    'controller': ctrl_name,
                                    'scenario': scenario,
                                    'algorithm': label
                                })
                        global_csv_data.append({
                            'seed': seed,
                            'generation': 'best',
                            'individual': 'best',
                            'fitness': best_fitness,
                            'controller': ctrl_name,
                            'scenario': scenario,
                            'algorithm': label,
                            'best_robot': json.dumps(best_robot.tolist())
                        })

    df_phase4 = pd.DataFrame(global_csv_data)
    df_phase4.to_csv("csvs/fase1/phase4/phase4.csv", index=False)

def plot_phase4_hyper_boxplots(csv_path="csvs/fase1/phase4/phase4.csv"):
    """ Plot boxplots comparing hyperparameter combinations for each scenario. """
    os.makedirs("plots/fase1/phase4/boxplots", exist_ok=True)

    df = pd.read_csv(csv_path)

    for scenario in df['scenario'].unique():
        df_scenario = df[df['scenario'] == scenario]
        best_entries = df_scenario[df_scenario['generation'] == 'best']

        labels = []
        data = []
        for alg in best_entries['algorithm'].unique():
            subset = best_entries[best_entries['algorithm'] == alg]['fitness'].values
            if len(subset) > 0:
                labels.append(alg.replace("GA_Hyper_", "").replace(f"_{scenario}", ""))
                data.append(subset)

        def sort_key(label):
            parts = label.split("_")
            return tuple(map(float, parts))
        sorted_indices = sorted(range(len(labels)), key=lambda i: sort_key(labels[i]))
        labels = [labels[i] for i in sorted_indices]
        data = [data[i] for i in sorted_indices]

        plt.figure(figsize=(18, 6))
        formatted_labels = []
        for l in labels:
            parts = l.split("_")
            if len(parts) == 3:
                formatted_labels.append(f"crossover={parts[0]}, mutation={parts[1]}, elitism={parts[2]}")
            else:
                formatted_labels.append(l)
        box = plt.boxplot(data, patch_artist=True, tick_labels=formatted_labels)
        colors = plt.cm.Set3.colors
        for patch, color in zip(box['boxes'], colors * (len(data) // len(colors) + 1)):
            patch.set_facecolor(color)

        means = [np.mean(d) for d in data]
        for i, mean in enumerate(means):
            plt.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=9)

        plt.title(f"Phase 4 Hyperparameter Tuning - {scenario}")
        plt.ylabel("Best Fitness per Seed")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/fase1/phase4/boxplots/phase4_boxplot_{scenario}.png")
        plt.show()

def extract_hyperparameters(alg_name, scenario):
    """Helper to extract float values of (cr, mr, er) from algorithm name."""
    clean = alg_name.replace("GA_Hyper_", "").replace(f"_{scenario}", "")
    try:
        cr, mr, er = map(float, clean.split("_"))
        return cr, mr, er
    except ValueError:
        return 999, 999, 999

def plot_phase4_hyper_curves(csv_path="csvs/fase1/phase4/phase4.csv"):
    """Plot average fitness evolution for all hyperparameter settings, ordered by (crossover, mutation, elitism)."""

    os.makedirs("plots/fase1/phase4/average_curves", exist_ok=True)

    df = pd.read_csv(csv_path)

    for scenario in df['scenario'].unique():
        df_scenario = df[(df['scenario'] == scenario) & (df['generation'] != 'best')].copy()
        df_scenario['generation'] = df_scenario['generation'].astype(int)

        algs = df_scenario['algorithm'].unique()
        sorted_algs = sorted(algs, key=lambda a: extract_hyperparameters(a, scenario))

        plt.figure(figsize=(18, 8))
        for alg in sorted_algs:
            cr, mr, er = extract_hyperparameters(alg, scenario)
            label = f"crossover={cr}, mutation={mr}, elitism={er}"
            df_alg = df_scenario[df_scenario['algorithm'] == alg]
            mean_fitness = df_alg.groupby('generation')['fitness'].mean()
            plt.plot(mean_fitness.index, mean_fitness.values, label=label, linewidth=2.0)

        plt.title(f"Mean Fitness Evolution - Phase 4 Hyperparameters ({scenario})", fontsize=15, fontweight='bold')
        plt.xlabel("Generation")
        plt.ylabel("Mean Fitness")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"plots/fase1/phase4/average_curves/phase4_average_curve_{scenario}.png")
        plt.show()

def run_phase4_statistics(csv_path="csvs/fase1/phase4/phase4.csv"):
    """
    Perform statistical comparison of hyperparameter tuning results from phase4.csv.
    For each scenario:
      - Tests normality.
      - Applies ANOVA or Kruskal-Wallis.
      - Identifies the best setting by mean fitness.
      - Saves statistical results and best hyperparameters into separate CSVs.
    """
    os.makedirs("csvs/fase1/phase4", exist_ok=True)

    df = pd.read_csv(csv_path)
    scenario_set = df['scenario'].unique()

    for scenario in scenario_set:
        df_scenario = df[df['scenario'] == scenario]
        df_scenario = df_scenario[df_scenario['generation'] == 'best']

        if df_scenario.empty:
            print(f"No hyperparameter data for scenario {scenario}. Skipping.")
            continue

        grouped = df_scenario.groupby(['algorithm', 'seed'])['fitness'].max().reset_index()
        algorithms = grouped['algorithm'].unique()
        data_per_algorithm = [grouped[grouped['algorithm'] == alg]['fitness'].values for alg in algorithms]

        normal = True
        for data in data_per_algorithm:
            if len(data) < 3:
                normal = False
                break
            stat, p = shapiro(data)
            if p < 0.05:
                normal = False
                break

        if normal:
            test_used = "ANOVA"
            stat, p_value = f_oneway(*data_per_algorithm)
        else:
            test_used = "Kruskal-Wallis"
            stat, p_value = kruskal(*data_per_algorithm)

        print(f"\nPhase 4 - Scenario: {scenario}")
        print(f"Test Used: {test_used}")
        print(f"Statistic={stat:.4f}, p-value={p_value:.4f}")

        significance = "Yes" if p_value < 0.05 else "No"

        result_data = {
            'Scenario': [scenario],
            'Test Used': [test_used],
            'Statistic': [stat],
            'p-value': [p_value],
            'Significant': [significance]
        }
        df_result = pd.DataFrame(result_data)
        df_result.to_csv(f"csvs/fase1/phase4/statistical_test_{scenario}.csv", index=False)

        mean_fitness_per_algorithm = grouped.groupby('algorithm')['fitness'].mean()
        best_algorithm = mean_fitness_per_algorithm.idxmax()

        clean = best_algorithm.replace("GA_Hyper_", "").replace(f"_{scenario}", "")
        best_crossover, best_mutation, best_elitism = map(float, clean.split("_"))

        best_hyper_data = {
            'scenario': [scenario],
            'best_crossover_rate': [best_crossover],
            'best_mutation_rate': [best_mutation],
            'best_elitism_rate': [best_elitism]
        }
        df_best = pd.DataFrame(best_hyper_data)
        df_best.to_csv(f"csvs/fase1/phase4/best_hyper_{scenario}.csv", index=False)

# MAIN FUNCTION
def plot_phase4_summary_tables():
    """ Summarize and display phase 4 results: statistical tests and best hyperparameter configurations. """
    os.makedirs("plots/fase1/phase4", exist_ok=True)

    test_files = glob.glob("csvs/fase1/phase4/statistical_test_*.csv")
    best_hyper_files = glob.glob("csvs/fase1/phase4/best_hyper_*.csv")

    df_tests = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
    df_best_hyper = pd.concat([pd.read_csv(f) for f in best_hyper_files], ignore_index=True)

    df_summary = df_tests.merge(df_best_hyper, left_on="Scenario", right_on="scenario")

    df_summary.to_csv("csvs/fase1/phase4/summary_phase4.csv", index=False)
    print("\nPhase 4 Statistical Summary")
    print(df_summary)

def main():
    controller_set = [
        (alternating_gait, "Alternating"),
        (sinusoidal_wave, "Sinusoidal"),
        (hopping_motion, "Hopping")
    ]
    scenario_set = ['Walker-v0', 'BridgeWalker-v0']
    
    print("\nPHASE 1: Controller Selection") 
    all_results_phase1, best_controllers = run_phase1(controller_set, scenario_set)
    run_stagnation_analysis("csvs/fase1/phase1.csv", controller_set, scenario_set)
    run_controller_comparison_boxplot("csvs/fase1/phase1.csv", controller_set, scenario_set)

    print("\nPHASE 2: Algorithm Comparison")
    all_results_phase2 = run_phase2(scenario_set, "csvs/fase1/phase1_best_controllers.csv")
    plot_phase2_comparison_boxplots("csvs/fase1/phase2/phase2.csv", scenario_set)
    plot_phase2_average_curves("csvs/fase1/phase2/phase2.csv", scenario_set)

    print("\nPHASE 3: Statistical Test and Best GA Selection")
    run_phase3("csvs/fase1/phase2/phase2.csv")
    plot_phase3_summary_tables()

    print("\nPHASE 4: Hyperparameter Tuning on Best GA")
    run_phase4(scenario_set)
    plot_phase4_hyper_boxplots()
    plot_phase4_hyper_curves()
    run_phase4_statistics()
    plot_phase4_summary_tables()

if __name__ == "__main__":
    main()
