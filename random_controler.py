import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import matplotlib.pyplot as plt
import os
import imageio
import copy
import pandas as pd
from scipy.stats import kruskal, f_oneway, shapiro
import glob
import itertools

# GLOBAL PARAMETERS
DEBUG_MODE = False 

SEEDS = [42] if DEBUG_MODE else [42, 123, 999, 2025, 7]
#NUM_GENERATIONS = 25 if DEBUG_MODE else 100
STEPS = 500

# Evolution Strategies Parameters
ES_MU = 20
ES_LAMBDA = 40
ES_MUTATION_STD_INIT = 0.3
ES_MUTATION_STD = ES_MUTATION_STD_INIT

# Local mutation probability for gaussian mutation
MUTATION_PROBABILITY = 0.2

# Differential Evolution Parameters
DE_POP_SIZE = 50
DE_MUTATION_FACTOR = 0.5
DE_CROSSOVER_RATE = 0.9

# Evaluation-aligned generations per algorithm
TOTAL_EVALUATIONS = 1000
NUM_GENERATIONS_RS = TOTAL_EVALUATIONS
NUM_GENERATIONS_ES = TOTAL_EVALUATIONS // ES_LAMBDA
NUM_GENERATIONS_ES_PLUS = TOTAL_EVALUATIONS // (ES_MU + ES_LAMBDA)
NUM_GENERATIONS_DE = TOTAL_EVALUATIONS // DE_POP_SIZE

SCENARIOS = ['DownStepper-v0', 'ObstacleTraverser-v0']
SCENARIO = SCENARIOS[0]

robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])

def init_env(scenario):
    connectivity = get_full_connectivity(robot_structure)
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    return env, sim, brain, connectivity

# FITNESS FUNCTION 
def evaluate_fitness(weights, view=False, scenario=None):
    env, sim, brain, connectivity = init_env(scenario or SCENARIO)
    set_weights(brain, weights)  # Load weights into the network
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    t_reward = 0
    for t in range(STEPS):  
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten() # Get action
        if view:
            viewer.render('screen') 
        state, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()
    return t_reward 

# RANDOM SEARCH ALGORITHM
def run_random_search(env, sim, brain, scenario):
    best_fitness = -np.inf
    best_weights = None

    fitness_history = []
    for generation in range(NUM_GENERATIONS_RS):
        # Generate random weights for the neural network
        random_weights = [np.random.randn(*param.shape) for param in brain.parameters()]
        
        # Evaluate the fitness of the current weights
        fitness = evaluate_fitness(random_weights, scenario=scenario)
        
        # Check if the current weights are the best so far
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = random_weights
        
        fitness_history.append(fitness)
        print(f"Generation {generation + 1}/{NUM_GENERATIONS_RS}, Fitness: {fitness}")

    # Set the best weights found
    set_weights(brain, best_weights)
    print(f"Best Fitness: {best_fitness}")
    return best_weights, fitness_history

# Mutation and Crossover Functions
def mutate_gaussian(weights, sigma):
    mutated_weights = []
    for w in copy.deepcopy(weights):
        w_mut = np.copy(w)
        mutation_mask = np.random.rand(*w.shape) < MUTATION_PROBABILITY
        noise = np.random.normal(0, sigma, size=w.shape)
        w_mut[mutation_mask] += noise[mutation_mask]
        mutated_weights.append(w_mut)
    return mutated_weights

def mutate_differential(x1, x2, x3, F):
    return [x1_i + F * (x2_i - x3_i) for x1_i, x2_i, x3_i in zip(copy.deepcopy(x1), x2, x3)]

def mutate_differential_best2(best, x1, x2, x3, x4, F):
    return [best_i + F * (x1_i - x2_i + x3_i - x4_i)
            for best_i, x1_i, x2_i, x3_i, x4_i in zip(best, x1, x2, x3, x4)]

def mutate_differential_rand2(x1, x2, x3, x4, x5, F):
    return [x1_l + F * (x2_l - x3_l) + F * (x4_l - x5_l)
            for x1_l, x2_l, x3_l, x4_l, x5_l in zip(x1, x2, x3, x4, x5)]

def crossover_bin(target, mutant, CR):
    trial = []
    for t_layer, m_layer in zip(target, mutant):
        shape = t_layer.shape
        flat_t = t_layer.flatten()
        flat_m = m_layer.flatten()
        flat_trial = np.copy(flat_t)
        rand_index = random.randint(0, flat_t.size - 1)
        for i in range(flat_t.size):
            if random.random() < CR or i == rand_index:
                flat_trial[i] = flat_m[i]
        trial.append(flat_trial.reshape(shape))
    return trial

# EVOLUTION STRATEGY ALGORITHMS
def run_evolution_strategy(env, sim, brain, scenario):
    global ES_MUTATION_STD
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    # Initialize parent population with random weights
    parents = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(ES_MU) ]

    for gen in range(NUM_GENERATIONS_ES):
        offspring = []
        # Generate offspring
        for _ in range(ES_LAMBDA):
            parent = random.choice(parents)
            child = mutate_gaussian(parent, ES_MUTATION_STD)
            offspring.append(child)

        offspring_fitness = [evaluate_fitness(ind, scenario=scenario) for ind in offspring]
        # Select top ES_MU offspring based on fitness to become new parents
        selected_indices = np.argsort(offspring_fitness)[-ES_MU:]
        parents = [offspring[i] for i in selected_indices]

        # Track and update the best individual 
        gen_best_fitness = max(offspring_fitness)
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = offspring[np.argmax(offspring_fitness)]

        fitness_history.append(best_fitness)
        print(f"[ES] Generation {gen+1}/{NUM_GENERATIONS_ES} | Best Fitness: {best_fitness:.2f}")

        # Adaptive mutation logic
        if gen > 0 and fitness_history[-1] > fitness_history[-2]:
            ES_MUTATION_STD *= 0.95 
        else:
            ES_MUTATION_STD *= 1.05 

    set_weights(brain, best_individual)
    return best_individual, fitness_history

def run_evolution_strategy_plus(env, sim, brain, scenario):
    global ES_MUTATION_STD
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    # Initialize parent population with random weights
    parents = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(ES_MU) ]

    for gen in range(NUM_GENERATIONS_ES_PLUS):
        offspring = []
        # Generate offspring 
        for _ in range(ES_LAMBDA):
            parent = random.choice(parents)
            child = mutate_gaussian(parent, ES_MUTATION_STD)
            offspring.append(child)

        # Combine parents and offspring before selection (ES+)
        combined = parents + offspring
        combined_fitness = [evaluate_fitness(ind, scenario=scenario) for ind in combined]
        # Select top ES_MU individuals from combined set based on fitness
        selected_indices = np.argsort(combined_fitness)[-ES_MU:]
        parents = [combined[i] for i in selected_indices]

        # Track and update the best individual
        gen_best_fitness = max(combined_fitness)
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = combined[np.argmax(combined_fitness)]

        fitness_history.append(best_fitness)
        print(f"[ES+] Generation {gen+1}/{NUM_GENERATIONS_ES_PLUS} | Best Fitness: {best_fitness:.2f}")

        # Adaptive mutation
        if gen > 0 and fitness_history[-1] > fitness_history[-2]:
            ES_MUTATION_STD *= 0.95 
        else:
            ES_MUTATION_STD *= 1.05

    set_weights(brain, best_individual)
    return best_individual, fitness_history

# DIFFERENTIAL EVOLUTION ALGORITHM 
def run_differential_evolution(env, sim, brain, scenario):
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    # Initialize population with random weights
    population = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(DE_POP_SIZE) ]
    fitnesses = [evaluate_fitness(ind, scenario=scenario) for ind in population]

    for gen in range(NUM_GENERATIONS_DE):
        new_population = []
        for i in range(DE_POP_SIZE):
            # DE/rand/1 mutation: select three distinct individuals (excluding current)
            indices = list(range(DE_POP_SIZE))
            indices.remove(i)
            x1, x2, x3 = random.sample([population[j] for j in indices], 3)
            mutant = mutate_differential(x1, x2, x3, DE_MUTATION_FACTOR)
            target = population[i]
            # Binomial crossover
            trial = crossover_bin(target, mutant, DE_CROSSOVER_RATE)

            # Selection
            trial_fitness = evaluate_fitness(trial, scenario=scenario)
            if trial_fitness > fitnesses[i]:
                new_population.append(trial)
                fitnesses[i] = trial_fitness
                # Update best individual
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial
            else:
                new_population.append(target)

        population = new_population
        fitness_history.append(best_fitness)
        print(f"[DE_rand] Generation {gen+1}/{NUM_GENERATIONS_DE} | Best Fitness: {best_fitness:.2f}")

    set_weights(brain, best_individual)
    return best_individual, fitness_history

# DE_jDE VARIANT
def run_differential_evolution_jde(env, sim, brain, scenario):
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    tau1 = 0.1
    tau2 = 0.1

    # Initialize population with random weights, F and CR for jDE
    population = []
    for _ in range(DE_POP_SIZE):
        weights = [np.random.randn(*param.shape) for param in brain.parameters()]
        F = np.random.uniform(0.1, 1.0)
        CR = np.random.uniform(0.0, 1.0)
        population.append((weights, F, CR))

    fitnesses = [evaluate_fitness(ind[0], scenario=scenario) for ind in population]

    for gen in range(NUM_GENERATIONS_DE):
        new_population = []

        for i in range(DE_POP_SIZE):
            x, F, CR = population[i]

            # jDE: Adapt F and CR with probability tau1 and tau2
            if np.random.rand() < tau1:
                F = np.random.uniform(0.1, 1.0)
            if np.random.rand() < tau2:
                CR = np.random.uniform(0.0, 1.0)

            # DE/rand/1 mutation with individuals own F and CR
            indices = list(range(DE_POP_SIZE))
            indices.remove(i)
            x1, x2, x3 = random.sample([population[j][0] for j in indices], 3)

            mutant = mutate_differential(x1, x2, x3, F)
            # Binomial crossover
            trial = crossover_bin(x, mutant, CR)

            # Selection
            trial_fitness = evaluate_fitness(trial, scenario=scenario)
            if trial_fitness > fitnesses[i]:
                new_population.append((trial, F, CR))
                fitnesses[i] = trial_fitness
                # Update best individual 
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial
            else:
                new_population.append((x, F, CR))

        population = new_population
        fitness_history.append(best_fitness)
        print(f"[DE_jDE] Generation {gen+1}/{NUM_GENERATIONS_DE} | Best Fitness: {best_fitness:.2f}")

    set_weights(brain, best_individual)
    avg_f = np.mean([ind[1] for ind in population])
    avg_cr = np.mean([ind[2] for ind in population])
    return best_individual, fitness_history, avg_f, avg_cr

# DIFFERENTIAL EVOLUTION BEST/1/BIN ALGORITHM
def run_differential_evolution_best(env, sim, brain, scenario):
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    # Initialize population with random weights
    population = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(DE_POP_SIZE) ]
    fitnesses = [evaluate_fitness(ind, scenario=scenario) for ind in population]

    for gen in range(NUM_GENERATIONS_DE):
        new_population = []
        # Identify current best
        best_idx = np.argmax(fitnesses)
        best_individual_current = population[best_idx]

        for i in range(DE_POP_SIZE):
            # DE/best/1 mutation: best + F*(x2-x3)
            indices = list(range(DE_POP_SIZE))
            indices.remove(i)
            x2, x3 = random.sample([population[j] for j in indices], 2)
            mutant = mutate_differential(best_individual_current, x2, x3, DE_MUTATION_FACTOR)
            target = population[i]
            # Binomial crossover
            trial = crossover_bin(target, mutant, DE_CROSSOVER_RATE)

            # Selection
            trial_fitness = evaluate_fitness(trial, scenario=scenario)
            if trial_fitness > fitnesses[i]:
                new_population.append(trial)
                fitnesses[i] = trial_fitness
                # Update best individual if necessary
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial
            else:
                new_population.append(target)

        population = new_population
        fitness_history.append(best_fitness)
        print(f"[DE_best] Generation {gen+1}/{NUM_GENERATIONS_DE} | Best Fitness: {best_fitness:.2f}")

    set_weights(brain, best_individual)
    return best_individual, fitness_history

# DIFFERENTIAL EVOLUTION BEST/2/BIN ALGORITHM
def run_differential_evolution_best2(env, sim, brain, scenario):
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    # Initialize population with random weights
    population = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(DE_POP_SIZE) ]
    fitnesses = [evaluate_fitness(ind, scenario=scenario) for ind in population]

    for gen in range(NUM_GENERATIONS_DE):
        new_population = []
        # Identify current best
        best_idx = np.argmax(fitnesses)
        best_individual_current = population[best_idx]

        for i in range(DE_POP_SIZE):
            # DE/best/2 mutation: best + F*(x1-x2 + x3-x4)
            indices = list(range(DE_POP_SIZE))
            indices.remove(i)
            sample_indices = [j for j in indices if j != best_idx]
            x1, x2, x3, x4 = random.sample([population[j] for j in sample_indices], 4)
            mutant = mutate_differential_best2(best_individual_current, x1, x2, x3, x4, DE_MUTATION_FACTOR)
            target = population[i]
            # Binomial crossover
            trial = crossover_bin(target, mutant, DE_CROSSOVER_RATE)

            # Selection
            trial_fitness = evaluate_fitness(trial, scenario=scenario)
            if trial_fitness > fitnesses[i]:
                new_population.append(trial)
                fitnesses[i] = trial_fitness
                # Update best individuay
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial
            else:
                new_population.append(target)

        population = new_population
        fitness_history.append(best_fitness)
        print(f"[DE_best2] Generation {gen+1}/{NUM_GENERATIONS_DE} | Best Fitness: {best_fitness:.2f}")

    set_weights(brain, best_individual)
    return best_individual, fitness_history

# DIFFERENTIAL EVOLUTION RAND/2/BIN ALGORITHM
def run_differential_evolution_rand2(env, sim, brain, scenario):
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []

    # Initialize population with random weights
    population = [ [np.random.randn(*param.shape) for param in brain.parameters()] for _ in range(DE_POP_SIZE) ]
    fitnesses = [evaluate_fitness(ind, scenario=scenario) for ind in population]

    for gen in range(NUM_GENERATIONS_DE):
        new_population = []
        for i in range(DE_POP_SIZE):
            # DE/rand/2 mutation: x1 + F*(x2-x3 + x4-x5)
            indices = list(range(DE_POP_SIZE))
            indices.remove(i)
            x1, x2, x3, x4, x5 = random.sample([population[j] for j in indices], 5)
            mutant = mutate_differential_rand2(x1, x2, x3, x4, x5, DE_MUTATION_FACTOR)
            target = population[i]
            # Binomial crossover
            trial = crossover_bin(target, mutant, DE_CROSSOVER_RATE)

            # Selection
            trial_fitness = evaluate_fitness(trial, scenario=scenario)
            if trial_fitness > fitnesses[i]:
                new_population.append(trial)
                fitnesses[i] = trial_fitness
                # Update best individual
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_individual = trial
            else:
                new_population.append(target)

        population = new_population
        fitness_history.append(best_fitness)
        print(f"[DE_rand2] Generation {gen+1}/{NUM_GENERATIONS_DE} | Best Fitness: {best_fitness:.2f}")

    set_weights(brain, best_individual)
    return best_individual, fitness_history

# VISUALIZATION
def visualize_policy(weights, scenario, algorithm="Unknown"):
    env, sim, brain, connectivity = init_env(scenario)
    set_weights(brain, weights)
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    state = env.reset()[0]
    frames = []

    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        frames.append(viewer.render('rgb_array'))
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    viewer.close()
    env.close()

    save_policy_gif(frames, scenario, algorithm)

def save_policy_gif(frames, scenario, algorithm="Unknown"):
    folder = os.path.join("gifs", "fase2", scenario, algorithm)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{algorithm}_{scenario}.gif")
    imageio.mimsave(path, frames, duration=0.01, optimize=True)

# PHASE 1 FUNCTIONS
def plot_phase1_mean_curves():
    """ Plot mean fitness evolution curves from individual algorithm CSVs. """
    for scenario in SCENARIOS:
        plt.figure(figsize=(12, 7))
        folder = os.path.join("csvs", "fase2", scenario)
        algorithms = ["ES", "ES_PLUS", "DE_rand", "DE_best"]
        colors = {'ES': 'blue', 'ES_PLUS': 'green', 'DE_rand': 'orange', 'DE_best': 'purple'}

        for algorithm in algorithms:
            path = os.path.join(folder, f"{algorithm}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                mean_fitness_per_gen = df.mean(axis=1)
                plt.plot(range(1, len(mean_fitness_per_gen)+1), mean_fitness_per_gen,
                         label=algorithm, color=colors.get(algorithm, 'black'))

        plt.title(f"Phase 1 - Mean Fitness Evolution (Without RS) - {scenario}")
        plt.xlabel("Generation")
        plt.ylabel("Mean Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_folder = os.path.join("plots", "fase2", scenario, "mean_curves")
        os.makedirs(plot_folder, exist_ok=True)
        plt.savefig(os.path.join(plot_folder, f"mean_curve_{scenario}_no_rs.png"))
        plt.close()

        path = os.path.join(folder, "RS.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            mean_fitness_per_gen = df.mean(axis=1)
            plt.figure(figsize=(12, 7))
            plt.plot(range(1, len(mean_fitness_per_gen)+1), mean_fitness_per_gen,
                     label="RS", color='red')

            plt.title(f"Phase 1 - Random Search Fitness Evolution - {scenario}")
            plt.xlabel("Generation")
            plt.ylabel("Mean Fitness")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f"mean_curve_{scenario}_RS_only.png"))
            plt.close()

def plot_phase1_boxplots():
    df_summary = pd.read_csv("csvs/fase2/summary_phase1.csv")
    box_colors = {'ES': 'lightblue', 'ES_PLUS': 'lightgreen', 'DE_rand': 'lightyellow', 'DE_best': 'violet', 'RS': 'lightcoral'}
    for scenario in df_summary["Scenario"].unique():
        plt.figure()
        subset = df_summary[df_summary["Scenario"] == scenario]
        labels = subset["Algorithm"].unique()
        data = [subset[subset["Algorithm"] == algo]["FinalFitness"].values for algo in labels]

        box = plt.boxplot(data, patch_artist=True, tick_labels=labels)
        for patch, algo in zip(box['boxes'], labels):
            patch.set_facecolor(box_colors.get(algo, 'lightgrey'))

        means = [np.mean(values) for values in data]
        for i, mean in enumerate(means):
            plt.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=9)

        plt.title(f"Algorithm Comparison - {scenario}")
        plt.ylabel("Final Fitness")
        plt.grid(True)
        plt.tight_layout()

        folder = os.path.join("plots", "fase2", "boxplots")
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"boxplot_{scenario}.png"))
        plt.close()

def run_phase1():
    all_records = []

    for scenario in SCENARIOS:
        for algorithm, run_function in [
            ("ES", run_evolution_strategy),
            ("ES_PLUS", run_evolution_strategy_plus),
            ("DE_rand", run_differential_evolution),
            ("DE_best", run_differential_evolution_best),
            ("RS", run_random_search)
        ]:
            fitness_per_seed = []
            for seed in SEEDS:
                print(f"\n{scenario} - {algorithm} - Seed {seed}")
                env, sim, brain, connectivity = init_env(scenario)
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                if "ES" in algorithm:
                    global ES_MUTATION_STD
                    ES_MUTATION_STD = ES_MUTATION_STD_INIT
                best_weights, fitness_history = run_function(env, sim, brain, scenario)
                visualize_policy(best_weights, scenario, algorithm=algorithm)
                fitness_per_seed.append(fitness_history)
                final_fitness = fitness_history[-1]
                all_records.append({"Scenario": scenario, "Algorithm": algorithm, "Seed": seed, "FinalFitness": final_fitness})

            folder = os.path.join("csvs", "fase2", scenario)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{algorithm}.csv")
            pd.DataFrame(fitness_per_seed).T.to_csv(path, index=False)

    summary_folder = os.path.join("csvs", "fase2")
    os.makedirs(summary_folder, exist_ok=True)
    df_summary = pd.DataFrame(all_records)
    df_summary.to_csv(os.path.join(summary_folder, "summary_phase1.csv"), index=False)

# PHASE 2 FUNCTIONS
def run_phase2(csv_path="csvs/fase2/summary_phase1.csv"):
    """ Perform statistical tests across algorithms to select the best controller optimizer (phase2 in phase2). """
    os.makedirs("csvs/fase2/phase2", exist_ok=True)

    df = pd.read_csv(csv_path)
    scenario_set = df['Scenario'].unique()

    for scenario in scenario_set:
        df_scenario = df[df['Scenario'] == scenario]

        df_scenario['Algorithm'] = df_scenario['Algorithm'].replace("DE", "DE_rand")
        algorithms = df_scenario['Algorithm'].unique()
        data_per_algorithm = [df_scenario[df_scenario['Algorithm'] == alg]['FinalFitness'].values for alg in algorithms]

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

        significance = "Yes" if p_value < 0.05 else "No"

        print(f"\nScenario: {scenario}")
        print(f"Test Used: {test_used}")
        print(f"Statistic={stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            print("There are statistically significant differences between algorithms.")
        else:
            print("There are no statistically significant differences between algorithms.")

        result_data = {
            'Scenario': [scenario],
            'Test Used': [test_used],
            'Statistic': [stat],
            'p-value': [p_value],
            'Significant': [significance]
        }
        df_result = pd.DataFrame(result_data)
        df_result.to_csv(f"csvs/fase2/phase2/statistical_test_{scenario}.csv", index=False)

        mean_fitness_per_algorithm = df_scenario.groupby('Algorithm')['FinalFitness'].mean()
        best_algorithm = mean_fitness_per_algorithm.idxmax()
        if best_algorithm == "DE":
            best_algorithm = "DE_rand"

        best_algo_data = {
            'Scenario': [scenario],
            'Best Algorithm': [best_algorithm]
        }
        df_best_algo = pd.DataFrame(best_algo_data)
        df_best_algo.to_csv(f"csvs/fase2/phase2/best_algorithm_{scenario}.csv", index=False)

def plot_phase2_summary_tables():
    """ Summarize and display phase2 results: statistical tests and best algorithms. """
    os.makedirs("plots/fase2/phase2", exist_ok=True)

    test_files = glob.glob("csvs/fase2/phase2/statistical_test_*.csv")
    best_algo_files = glob.glob("csvs/fase2/phase2/best_algorithm_*.csv")

    df_tests = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
    df_best_algos = pd.concat([pd.read_csv(f) for f in best_algo_files], ignore_index=True)

    df_summary = df_tests.merge(df_best_algos, on="Scenario")
    df_summary.to_csv("csvs/fase2/phase2/summary_phase2.csv", index=False)
    print("\nPhase 2 Statistical Summary")
    print(df_summary)

# PHASE 3 FUNCTIONS
def run_phase3(summary_phase2_path="csvs/fase2/phase2/summary_phase2.csv"):
    global ES_MUTATION_STD, DE_MUTATION_FACTOR, DE_CROSSOVER_RATE
    os.makedirs("csvs/fase2/phase3", exist_ok=True)

    df_phase2 = pd.read_csv(summary_phase2_path)

    all_records_per_generation = []

    for idx, row in df_phase2.iterrows():
        scenario = row["Scenario"]
        best_algorithm = row["Best Algorithm"]

        if best_algorithm == "DE":
            best_algorithm = "DE_rand"

        print(f"\nRunning hypertuning for {best_algorithm} on {scenario}...")

        best_config = None
        best_fitness = -np.inf
        best_weights = None
        all_records = []

        if best_algorithm.startswith("ES"):
            sigma_values = [0.1, 0.2, 0.3, 0.4]
            for sigma in sigma_values:
                fitnesses = []
                best_weights_this_config = None
                best_fitness_this_config = -np.inf
                for seed in SEEDS:
                    print(f"\nRunning ES with sigma={sigma}, Seed={seed}")
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    ES_MUTATION_STD = sigma
                    env, sim, brain, connectivity = init_env(scenario)
                    if best_algorithm == "ES":
                        weights, fitness_history = run_evolution_strategy(env, sim, brain, scenario)
                    else:
                        weights, fitness_history = run_evolution_strategy_plus(env, sim, brain, scenario)
                    fitnesses.append(fitness_history[-1])
                    if fitness_history[-1] > best_fitness_this_config:
                        best_fitness_this_config = fitness_history[-1]
                        best_weights_this_config = weights
                    for generation_idx, fitness in enumerate(fitness_history, start=1):
                        all_records_per_generation.append({
                            'Scenario': scenario,
                            'HyperConfig': f"sigma_{sigma}",
                            'Seed': seed,
                            'Generation': generation_idx,
                            'Fitness': fitness
                        })

                avg_fitness = np.mean(fitnesses)
                all_records.append({"HyperConfig": f"sigma_{sigma}", "FinalFitness": avg_fitness})
                if avg_fitness > best_fitness:
                    best_fitness = avg_fitness
                    best_config = {"sigma": sigma}
                    best_weights = best_weights_this_config

        elif best_algorithm in ["DE_rand", "DE_best"]:
            f_values = [0.3, 0.5, 0.7]
            cr_values = [0.5, 0.7, 0.9]

            if best_algorithm == "DE_rand":
                variant_funcs = [
                    ("DE_rand", run_differential_evolution),
                    ("DE_rand2", run_differential_evolution_rand2),
                    ("DE_rand_jDE", run_differential_evolution_jde),
                ]
            else:
                variant_funcs = [
                    ("DE_best", run_differential_evolution_best),
                    ("DE_best2", run_differential_evolution_best2),
                ]

            for de_variant, de_function in variant_funcs:
                if de_variant == "DE_rand_jDE":
                    f_cr_pairs = [(0.3, 0.5)] 
                else:
                    f_cr_pairs = [(f, cr) for f in f_values for cr in cr_values]

                for f, cr in f_cr_pairs:
                    fitnesses = []
                    best_weights_this_config = None
                    best_fitness_this_config = -np.inf
                    avg_f_list = []
                    avg_cr_list = []
                    for seed in SEEDS:
                        print(f"\nRunning {de_variant} with F={f}, CR={cr}, Seed={seed}")
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.manual_seed(seed)
                        DE_MUTATION_FACTOR = f
                        DE_CROSSOVER_RATE = cr
                        env, sim, brain, connectivity = init_env(scenario)
                        if de_variant == "DE_rand_jDE":
                            weights, fitness_history, avg_f, avg_cr = de_function(env, sim, brain, scenario)
                            avg_f_list.append(avg_f)
                            avg_cr_list.append(avg_cr)
                        else:
                            weights, fitness_history = de_function(env, sim, brain, scenario)
                        fitnesses.append(fitness_history[-1])
                        if fitness_history[-1] > best_fitness_this_config:
                            best_fitness_this_config = fitness_history[-1]
                            best_weights_this_config = weights
                        if de_variant == "DE_rand_jDE":
                            for generation_idx, fitness in enumerate(fitness_history, start=1):
                                all_records_per_generation.append({
                                    'Scenario': scenario,
                                    'HyperConfig': f"{de_variant}_F_{avg_f:.2f}_CR_{avg_cr:.2f}",
                                    'Seed': seed,
                                    'Generation': generation_idx,
                                    'Fitness': fitness
                                })
                        else:
                            for generation_idx, fitness in enumerate(fitness_history, start=1):
                                all_records_per_generation.append({
                                    'Scenario': scenario,
                                    'HyperConfig': f"{de_variant}_F_{f}_CR_{cr}",
                                    'Seed': seed,
                                    'Generation': generation_idx,
                                    'Fitness': fitness
                                })

                    avg_fitness = np.mean(fitnesses)
                    if de_variant == "DE_rand_jDE":
                        mean_f = np.mean(avg_f_list) if avg_f_list else 0
                        mean_cr = np.mean(avg_cr_list) if avg_cr_list else 0
                        all_records.append({"HyperConfig": f"{de_variant}_F_{mean_f:.2f}_CR_{mean_cr:.2f}", "FinalFitness": avg_fitness})
                        if avg_fitness > best_fitness:
                            best_fitness = avg_fitness
                            best_config = {"Variant": de_variant, "F": mean_f, "CR": mean_cr}
                            best_weights = best_weights_this_config
                    else:
                        all_records.append({"HyperConfig": f"{de_variant}_F_{f}_CR_{cr}", "FinalFitness": avg_fitness})
                        if avg_fitness > best_fitness:
                            best_fitness = avg_fitness
                            best_config = {"Variant": de_variant, "F": f, "CR": cr}
                            best_weights = best_weights_this_config

        df_all = pd.DataFrame(all_records)
        df_all.to_csv(f"csvs/fase2/phase3/hypertuning_{scenario}.csv", index=False)

        best_config_data = {
            'Scenario': [scenario],
            'Best_HyperConfig': [best_config]
        }
        pd.DataFrame(best_config_data).to_csv(f"csvs/fase2/phase3/best_hyperparameter_config_{scenario}.csv", index=False)

        folder = os.path.join("models", "fase2", "phase3", scenario)
        os.makedirs(folder, exist_ok=True)
        if best_weights is not None:
            np.savez(os.path.join(folder, f"best_weights_{scenario}.npz"), *best_weights)

    df_generations = pd.DataFrame(all_records_per_generation)
    df_generations.to_csv("csvs/fase2/phase3/phase3_generations.csv", index=False)

def plot_phase3_hyper_curves(csv_path="csvs/fase2/phase3/phase3_generations.csv"):
    """
    Plot mean fitness evolution curves per hyperparameter configuration, grouped by variant (DE_rand, DE_rand2, DE_best, ES, ES_PLUS).
    Each variant gets a separate subplot for clarity.
    """
    os.makedirs("plots/fase2/phase3/average_curves", exist_ok=True)

    df = pd.read_csv(csv_path)

    for scenario in df['Scenario'].unique():
        df_scenario = df[df['Scenario'] == scenario].copy()
        df_scenario["Variant"] = df_scenario["HyperConfig"].apply(
            lambda x: "_".join(x.split("_")[:2])
        )
        for variant in df_scenario["Variant"].unique():
            df_variant = df_scenario[df_scenario["Variant"] == variant]
            plt.figure(figsize=(14, 7))
            for config in df_variant["HyperConfig"].unique():
                df_config = df_variant[df_variant["HyperConfig"] == config]
                mean_fitness_per_gen = df_config.groupby('Generation')['Fitness'].mean()
                plt.plot(mean_fitness_per_gen.index, mean_fitness_per_gen.values, label=config, linewidth=1.5)

            plt.title(f"{scenario} - Mean Fitness Evolution - {variant}")
            plt.xlabel("Generation")
            plt.ylabel("Mean Fitness")
            plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/fase2/phase3/average_curves/phase3_average_curve_{scenario}_{variant}.png")
            plt.close()

def plot_phase3_summary_tables():
    """ Summarize and display phase3 results: best hyperparameters per scenario. """
    os.makedirs("plots/fase2/phase3", exist_ok=True)

    best_config_files = glob.glob("csvs/fase2/phase3/best_hyperparameter_config_*.csv")

    df_best_configs = pd.concat([pd.read_csv(f) for f in best_config_files], ignore_index=True)

    df_best_configs.to_csv("csvs/fase2/phase3/summary_phase3.csv", index=False)
    print("\nPhase 3 Hypertuning Summary")
    print(df_best_configs)

def run_phase3_statistics(csv_path="csvs/fase2/phase3/phase3_generations.csv"):
    """
    Run statistical tests (Shapiro-Wilk + ANOVA/Kruskal) for Phase 3 hyperparameter configurations.
    One test per scenario and variant.
    """
    os.makedirs("csvs/fase2/phase3/stats", exist_ok=True)
    df = pd.read_csv(csv_path)

    for scenario in df["Scenario"].unique():
        df_scenario = df[df["Scenario"] == scenario].copy()
        df_scenario["Variant"] = df_scenario["HyperConfig"].apply(lambda x: "_".join(x.split("_")[:2]))        
        for variant in df_scenario["Variant"].unique():
            df_variant = df_scenario[df_scenario["Variant"] == variant]
            final_gen_data = (
                df_variant.groupby(['HyperConfig', 'Seed'])
                .agg({'Generation': 'max'})
                .reset_index()
                .merge(df_variant, on=['HyperConfig', 'Seed', 'Generation'])
            )
            data_groups = [final_gen_data[final_gen_data['HyperConfig'] == label]["Fitness"].values for label in sorted(final_gen_data["HyperConfig"].unique())]

            if len(data_groups) < 2:
                print(f"\nPhase 3 - Scenario: {scenario} | Variant: {variant}")
                if variant == "DE_jDE":
                    print("Skipped statistical test: Only one configuration available (DE_jDE is an adaptive DE variant with no hyperparameter tuning).")
                else:
                    print("Skipped statistical test: Only one configuration available.")
                continue

            normal = True
            for data in data_groups:
                if len(data) >= 3:
                    stat, p = shapiro(data)
                    if p < 0.05:
                        normal = False
                        break
                else:
                    normal = False
                    break

            if normal:
                test_used = "ANOVA"
                stat, p_value = f_oneway(*data_groups)
            else:
                test_used = "Kruskal-Wallis"
                stat, p_value = kruskal(*data_groups)

            significance = "Yes" if p_value < 0.05 else "No"

            print(f"\nPhase 3 - Scenario: {scenario} | Variant: {variant}")
            print(f"Test Used: {test_used} | Statistic={stat:.4f} | p-value={p_value:.4f} | Significant: {significance}")

            result = pd.DataFrame([{
                "Scenario": scenario,
                "Variant": variant,
                "Test Used": test_used,
                "Statistic": stat,
                "p-value": p_value,
                "Significant": significance
            }])
            result.to_csv(f"csvs/fase2/phase3/stats/stat_test_{scenario}_{variant}.csv", index=False)

    print("\nPhase 3 Statistical Analysis Complete.")

def plot_phase3_hyper_boxplots(csv_path="csvs/fase2/phase3/phase3_generations.csv"):
    """ Plot boxplots comparing hyperparameter configurations using best fitness per seed. """
    os.makedirs("plots/fase2/phase3/boxplots", exist_ok=True)
    df = pd.read_csv(csv_path)

    for scenario in df["Scenario"].unique():
        df_scenario = df[df["Scenario"] == scenario]
        df_scenario = df_scenario.copy()
        df_scenario["Variant"] = df_scenario["HyperConfig"].apply(
            lambda x: "_".join(x.split("_")[:2])
        )
        for variant in df_scenario["Variant"].unique():
            plt.figure()
            df_variant = df_scenario[df_scenario["Variant"] == variant]

            final_gen_data = (
                df_variant.groupby(['HyperConfig', 'Seed'])
                .agg({'Generation': 'max'})
                .reset_index()
                .merge(df_variant, on=['HyperConfig', 'Seed', 'Generation'])
            )

            labels = sorted(final_gen_data['HyperConfig'].unique())
            data = [final_gen_data[final_gen_data['HyperConfig'] == label]["Fitness"].values for label in labels]

            box = plt.boxplot(data, patch_artist=True, tick_labels=labels)
            colors = plt.cm.Set3.colors
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            means = [np.mean(values) for values in data]
            for i, mean in enumerate(means):
                plt.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=9)

            plt.title(f"Phase 3 Hyperparameter Tuning - {scenario} ({variant})")
            plt.ylabel("Best Fitness")
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/fase2/phase3/boxplots/phase3_boxplot_{scenario}_{variant}.png")
            plt.close()

def visualize_phase3_best_weights(summary_csv="csvs/fase2/phase3/summary_phase3.csv"):
    """
    Generate and save policy visualizations (GIFs) for the best configuration of each scenario in Phase 3.
    """
    df = pd.read_csv(summary_csv)

    for _, row in df.iterrows():
        scenario = row["Scenario"]
        config = eval(str(row["Best_HyperConfig"]))
        variant = config["Variant"] if isinstance(config, dict) and "Variant" in config else "Unknown"

        weights_path = f"models/fase2/phase3/{scenario}/best_weights_{scenario}.npz"
        if os.path.exists(weights_path):
            loaded = np.load(weights_path)
            weights = [loaded[key] for key in loaded]
            print(f"Visualizing: {scenario} | {variant}")
            visualize_policy(weights, scenario, algorithm=variant)
        else:
            print(f"Warning: Weights not found for {scenario} | {variant}")

# PHASE 4 FUNCTIONS
def run_phase4(summary_phase3_path="csvs/fase2/phase3/summary_phase3.csv"):
    global DE_POP_SIZE, NUM_GENERATIONS_DE
    os.makedirs("csvs/fase2/phase4", exist_ok=True)

    df_summary = pd.read_csv(summary_phase3_path)
    all_records = []
    all_generations = []

    pop_sizes = [30, 50, 70]
    generations = [20, 40, 60]

    for idx, row in df_summary.iterrows():
        scenario = row["Scenario"]
        config = eval(str(row["Best_HyperConfig"]))
        variant = config["Variant"]

        for pop_size in pop_sizes:
            for gens in generations:
                DE_POP_SIZE = pop_size
                NUM_GENERATIONS_DE = gens
                fitnesses = []
                best_fitness_config = -np.inf
                best_weights = None
                for seed in SEEDS:
                    print(f"\n{scenario} | {variant} | POP_SIZE={pop_size} | GENS={gens} | Seed={seed}")
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    env, sim, brain, connectivity = init_env(scenario)
                    if variant == "DE_best2":
                        weights, fitness_history = run_differential_evolution_best2(env, sim, brain, scenario)
                    elif variant == "DE_jDE":
                        weights, fitness_history = run_differential_evolution_jde(env, sim, brain, scenario)
                    elif variant == "DE_rand":
                        weights, fitness_history = run_differential_evolution(env, sim, brain, scenario)
                    elif variant == "DE_rand2":
                        weights, fitness_history = run_differential_evolution_rand2(env, sim, brain, scenario)
                    elif variant == "DE_best":
                        weights, fitness_history = run_differential_evolution_best(env, sim, brain, scenario)
                    else:
                        print(f"Unsupported variant {variant}")
                        continue
                    fitnesses.append(fitness_history[-1])
                    if fitness_history[-1] > best_fitness_config:
                        best_weights = weights
                        best_fitness_config = fitness_history[-1]
                    for generation_idx, fitness in enumerate(fitness_history, start=1):
                        all_generations.append({
                            "Scenario": scenario,
                            "Variant": variant,
                            "POP_SIZE": pop_size,
                            "GENS": gens,
                            "Seed": seed,
                            "Generation": generation_idx,
                            "Fitness": fitness
                        })

                avg_fitness = np.mean(fitnesses)
                all_records.append({
                    "Scenario": scenario,
                    "Variant": variant,
                    "POP_SIZE": pop_size,
                    "GENS": gens,
                    "FinalFitness": avg_fitness
                })

                folder = os.path.join("models", "fase2", "phase4", scenario)
                os.makedirs(folder, exist_ok=True)
                if best_weights is not None:
                    np.savez(os.path.join(folder, f"best_weights_{variant}_P{pop_size}_G{gens}.npz"), *best_weights)

    pd.DataFrame(all_records).to_csv("csvs/fase2/phase4/phase4.csv", index=False)
    pd.DataFrame(all_generations).to_csv("csvs/fase2/phase4/phase4_generations.csv", index=False)

def plot_phase4_boxplots(csv_path="csvs/fase2/phase4/phase4_generations.csv"):
    """ Plot boxplots comparing POP/GENS configurations using best fitness per seed. """
    os.makedirs("plots/fase2/phase4/boxplots", exist_ok=True)
    df = pd.read_csv(csv_path)

    for scenario in df["Scenario"].unique():
        df_scenario = df[df["Scenario"] == scenario]
        df_scenario = df_scenario.copy()
        df_scenario["Config"] = df_scenario.apply(lambda row: f"{row['Variant']}_P{row['POP_SIZE']}_G{row['GENS']}", axis=1)

        final_gen_data = (
            df_scenario.groupby(['Config', 'Seed'])
            .agg({'Generation': 'max'})
            .reset_index()
            .merge(df_scenario, on=['Config', 'Seed', 'Generation'])
        )

        labels = sorted(final_gen_data['Config'].unique())
        data = [final_gen_data[final_gen_data['Config'] == label]["Fitness"].values for label in labels]

        plt.figure(figsize=(14, 6))
        box = plt.boxplot(data, patch_artist=True, tick_labels=labels)
        colors = plt.cm.Set3.colors
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        means = [np.mean(values) for values in data]
        for i, mean in enumerate(means):
            plt.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=9)

        plt.title(f"Phase 4 Hyperparameter Tuning - {scenario}")
        plt.ylabel("Best Fitness")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/fase2/phase4/boxplots/phase4_boxplot_{scenario}.png")
        plt.close()


def plot_phase4_curves(csv_path="csvs/fase2/phase4/phase4_generations.csv"):
    os.makedirs("plots/fase2/phase4/average_curves", exist_ok=True)
    df = pd.read_csv(csv_path)

    for scenario in df["Scenario"].unique():
        df_scenario = df[df["Scenario"] == scenario].copy()
        df_scenario["Variant"] = df_scenario["Variant"].astype(str)
        df_scenario["Label"] = df_scenario.apply(
            lambda row: f"{row['Variant']}_P{row['POP_SIZE']}_G{row['GENS']}", axis=1
        )

        for variant in df_scenario["Variant"].unique():
            df_variant = df_scenario[df_scenario["Variant"] == variant]
            labels = sorted(df_variant["Label"].unique())

            plt.figure(figsize=(14, 7))
            from matplotlib import cm
            colors = itertools.cycle(cm.tab10.colors)
            linestyles = itertools.cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))])
            style_cycle = itertools.cycle([(color, ls) for color in cm.tab10.colors for ls in ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]])
            for label in sorted(df_variant["Label"].unique()):
                df_config = df_variant[df_variant["Label"] == label]
                if df_config.empty:
                    print(f"[Warning] No data for {label}")
                    continue
                mean_curve = df_config.groupby("Generation")["Fitness"].mean()
                color, linestyle = next(style_cycle)
                plt.plot(mean_curve.index, mean_curve.values,
                         label=label, linewidth=2, linestyle=linestyle, color=color)

            plt.title(f"{scenario} - Mean Fitness Evolution - {variant}")
            plt.xlabel("Generation")
            plt.ylabel("Mean Fitness")
            handles, labels_ = plt.gca().get_legend_handles_labels()
            sorted_handles_labels = sorted(zip(handles, labels_), key=lambda x: x[1])
            handles, labels_ = zip(*sorted_handles_labels)
            plt.legend(handles, labels_, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/fase2/phase4/average_curves/phase4_average_curve_{scenario}_{variant}.png")
            plt.close()

def plot_phase4_summary_tables():
    """ Summarize and display phase4 results: best pop/gen combination per scenario. """
    os.makedirs("plots/fase2/phase4", exist_ok=True)

    df = pd.read_csv("csvs/fase2/phase4/phase4.csv")
    df_summary = df.loc[df.groupby("Scenario")["FinalFitness"].idxmax()].reset_index(drop=True)
    df_summary.to_csv("csvs/fase2/phase4/summary_phase4.csv", index=False)
    print("\nPhase 4 Final Configuration Summary")
    print(df_summary)

def run_phase4_statistics(csv_path="csvs/fase2/phase4/phase4_generations.csv"):
    """
    Run statistical tests (Shapiro-Wilk + ANOVA/Kruskal) for Phase 4 population/generation configurations.
    One test per scenario.
    """
    os.makedirs("csvs/fase2/phase4/stats", exist_ok=True)
    df = pd.read_csv(csv_path)

    for scenario in df["Scenario"].unique():
        df_scenario = df[df["Scenario"] == scenario].copy()
        df_scenario["Config"] = df_scenario.apply(lambda row: f"{row['Variant']}_P{row['POP_SIZE']}_G{row['GENS']}", axis=1)
        
        final_gen_data = (
            df_scenario.groupby(['Config', 'Seed'])
            .agg({'Generation': 'max'})
            .reset_index()
            .merge(df_scenario, on=['Config', 'Seed', 'Generation'])
        )

        data_groups = [final_gen_data[final_gen_data["Config"] == label]["Fitness"].values for label in sorted(final_gen_data["Config"].unique())]

        normal = True
        for data in data_groups:
            if len(data) >= 3:
                stat, p = shapiro(data)
                if p < 0.05:
                    normal = False
                    break
            else:
                normal = False
                break

        if normal:
            test_used = "ANOVA"
            stat, p_value = f_oneway(*data_groups)
        else:
            test_used = "Kruskal-Wallis"
            stat, p_value = kruskal(*data_groups)

        significance = "Yes" if p_value < 0.05 else "No"

        print(f"\nPhase 4 - Scenario: {scenario}")
        print(f"Test Used: {test_used} | Statistic={stat:.4f} | p-value={p_value:.4f} | Significant: {significance}")

        result = pd.DataFrame([{
            "Scenario": scenario,
            "Test Used": test_used,
            "Statistic": stat,
            "p-value": p_value,
            "Significant": significance
        }])
        result.to_csv(f"csvs/fase2/phase4/stats/stat_test_{scenario}.csv", index=False)

    print("\nPhase 4 Statistical Analysis Complete.")

def visualize_phase4_best_weights(summary_csv="csvs/fase2/phase4/summary_phase4.csv"):
    """
    Generate and save policy visualizations (GIFs) for the best configuration of each scenario in Phase 4.
    """
    df = pd.read_csv(summary_csv)

    for _, row in df.iterrows():
        scenario = row["Scenario"]
        variant = row["Variant"]
        pop_size = row["POP_SIZE"]
        gens = row["GENS"]
        algorithm_label = f"{variant}_P{pop_size}_G{gens}"

        weights_path = f"models/fase2/phase4/{scenario}/best_weights_{variant}_P{pop_size}_G{gens}.npz"
        if os.path.exists(weights_path):
            loaded = np.load(weights_path)
            weights = [loaded[key] for key in loaded]
            print(f"Visualizing: {scenario} | {algorithm_label}")
            visualize_policy(weights, scenario, algorithm=algorithm_label)
        else:
            print(f"Warning: Weights not found for {scenario} | {algorithm_label}")

# MAIN FUNCTION 
if __name__ == "__main__":
    print("Running Phase 1: Random Search, ES, ES+, DE_rand, DE_best")
    run_phase1()
    plot_phase1_mean_curves()
    plot_phase1_boxplots()

    print("\nRunning Phase 2: Statistical Tests and Best Algorithm Selection")
    run_phase2()
    plot_phase2_summary_tables()

    print("\nRunning Phase 3: Hyperparameter Tuning")
    run_phase3()
    plot_phase3_hyper_boxplots()
    plot_phase3_hyper_curves()
    run_phase3_statistics()
    plot_phase3_summary_tables()
    visualize_phase3_best_weights()

    print("\nRunning Phase 4: Final Tuning with best hyperparameters")
    run_phase4()
    plot_phase4_boxplots()
    plot_phase4_curves()
    run_phase4_statistics()
    plot_phase4_summary_tables()
    visualize_phase4_best_weights()    