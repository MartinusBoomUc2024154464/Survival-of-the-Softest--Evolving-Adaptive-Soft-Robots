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
import imageio

import itertools
import torch

import time

# GLOBAL PARAMETERS
SCENARIOS = ['GapJumper-v0', 'CaveCrawler-v0']
SCENARIO = SCENARIOS[0]
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]
DEBUG_MODE = False 

SEEDS = [42] if DEBUG_MODE else [42, 123, 999, 2025, 7]
STEPS = 500
POP_SIZE = 15 if DEBUG_MODE else 30
GENERATIONS = 20 if DEBUG_MODE else 50
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
ELISTIM = 0.1
MAX_ARCHIVE_SIZE = 10
N_PARTNERS = 5
MAX_STAGNATION = 10

def evaluate_fitness(structure, brain, scenario=None, view=False):
    try:
        env, sim, input_size, output_size, _ = init_env(structure, scenario or SCENARIO)
        if not hasattr(brain, "input_size") or not hasattr(brain, "output_size"):
            return -5.0
        if input_size != brain.input_size or output_size != brain.output_size:
            return -5.0
        if view:
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')
        state = env.reset()[0]
        t_reward = 0
        for t in range(STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = brain(state_tensor).detach().numpy().flatten()
            if len(action) != output_size:
                return -5.0
            if view:
                viewer.render('screen')
            state, reward, terminated, truncated, _ = env.step(action)
            t_reward += reward
            if terminated or truncated:
                break
        if view:
            viewer.close()
        env.close()
        return t_reward
    except Exception as e:
        print(f"[Evaluation ERROR] {e}")
        return -5.0
    
def generate_gif(structure, controller, scenario, seed):
    gif_path = f"gifs/fase3/phase1/{scenario}/seed_{seed}"
    os.makedirs(gif_path, exist_ok=True)
    try:
        env, sim, _, _, _ = init_env(structure, scenario)
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        state = env.reset()[0]
        frames = []
        for t in range(STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = controller(state_tensor).detach().numpy().flatten()
            frames.append(viewer.render('rgb_array'))
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        viewer.close()
        env.close()
        imageio.mimsave(os.path.join(gif_path, 'best.gif'), frames, duration=0.01)
    except Exception as e:
        print(f"[ERROR] Could not save gif for seed {seed} ({scenario}): {e}")

# STRUCTURE
def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def mutation_scramble(structure: np.ndarray) -> np.ndarray:
    structure = copy.deepcopy(structure)
    flat = structure.flatten()
    start, end = sorted(np.random.choice(len(flat), size=2, replace=False))
    scrambled = flat[start:end].copy()
    np.random.shuffle(scrambled)
    flat[start:end] = scrambled
    new_structure = flat.reshape(structure.shape)
    #print(f"[Mutation Scramble] Applied on structure: shape={structure.shape}")
    return new_structure

def mutate_structure(structure, MRATE=MUTATION_RATE):
    if np.random.rand() < MRATE:
        return mutation_scramble(structure)
    return copy.deepcopy(structure)

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)
    mask = np.random.rand(*parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    #print(f"[Crossover Uniform] Applied on parents with shape: {parent1.shape}")
    return child

def crossover_structure(parent1, parent2, CR=CROSSOVER_RATE):
    if np.random.rand() < CR:
        return crossover_uniform(parent1, parent2)
    return copy.deepcopy(random.choice([parent1, parent2]))


def average_fitness_per_individual(fitness_matrix, axis=0):
    if axis == 0:
        return [sum(row) / len(row) for row in fitness_matrix]
    else:
        return [sum(col) / len(col) for col in zip(*fitness_matrix)] 

def tournament_selection_from_list(population, fitnesses, k=TOURNAMENT_SIZE):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

# CONTROLLER
def init_env(robot_structure, scenario):
    connectivity = get_full_connectivity(robot_structure)
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    return env, sim, input_size, output_size, connectivity

def get_input_output_size(structure, scenario):
    env, _, input_size, output_size, _ = init_env(structure, scenario)
    env.close()
    return input_size, output_size

def create_random_controller(robot_structure, task):
    for attempt in range(20):
        try:
            env, sim, input_size, output_size, connectivity = init_env(robot_structure, task)
            controller = NeuralController(input_size, output_size)
            controller.input_size = input_size
            controller.output_size = output_size
            return controller, connectivity, env, sim
        except Exception as e:
            print(f"[Controller Creation Error] Attempt {attempt+1}: {e}")
            continue
    raise RuntimeError("Failed to create a valid controller after multiple attempts.")

def mutate_controller(controller, MRATE=MUTATION_RATE):
    new_controller = copy.deepcopy(controller)
    new_controller.input_size = controller.input_size
    new_controller.output_size = controller.output_size
    with torch.no_grad():
        for param in new_controller.parameters():
            if param.requires_grad:
                mutation_mask = torch.rand_like(param) < MRATE
                noise = torch.normal(mean=0.0, std=0.01, size=param.size())
                param.add_(mutation_mask * noise)
    #print(f"[Controller Mutation] Original: {brain_signature(controller)} | Mutated: {brain_signature(new_controller)}")
    return new_controller

def pad_or_crop_tensor(tensor, target_shape):
    current_shape = tensor.shape
    if len(target_shape) == 2:
        pad_rows = max(0, target_shape[0] - current_shape[0])
        pad_cols = max(0, target_shape[1] - current_shape[1])
        cropped = tensor[:target_shape[0], :target_shape[1]]
        padded = torch.nn.functional.pad(cropped, (0, pad_cols, 0, pad_rows))
        return padded
    elif len(target_shape) == 1:
        pad_size = max(0, target_shape[0] - current_shape[0])
        cropped = tensor[:target_shape[0]]
        padded = torch.nn.functional.pad(cropped, (0, pad_size))
        return padded
    else:
        return tensor 

def adapt_controller_to_structure(controller, new_structure, scenario):
    expected_input, expected_output = get_input_output_size(new_structure, scenario)
    new_brain = NeuralController(expected_input, expected_output)
    state1 = controller.state_dict()
    state2 = new_brain.state_dict()

    for key in state2:
        tensor1 = state1[key]
        padded = pad_or_crop_tensor(tensor1, state2[key].shape)
        state2[key] = padded

    new_brain.load_state_dict(state2)
    new_brain.input_size = expected_input
    new_brain.output_size = expected_output
    return new_brain

def crossover_controller(parent1, parent2, CR=CROSSOVER_RATE):
    if parent1.input_size != parent2.input_size or parent1.output_size != parent2.output_size:
        return copy.deepcopy(random.choice([parent1, parent2]))

    input_size = min(parent1.input_size, parent2.input_size)
    output_size = min(parent1.output_size, parent2.output_size)

    child = NeuralController(input_size, output_size)
    state1 = parent1.state_dict()
    state2 = parent2.state_dict()
    child_state = child.state_dict()

    for key in child_state:
        tensor1 = state1[key]
        tensor2 = state2[key]

        padded1 = pad_or_crop_tensor(tensor1, child_state[key].shape)
        padded2 = pad_or_crop_tensor(tensor2, child_state[key].shape)

        mask = torch.rand_like(padded1) < CR
        rand_index = torch.randint(0, padded1.numel(), (1,))
        mask.view(-1)[rand_index] = True
        new_tensor = torch.where(mask, padded2, padded1)
        child_state[key] = new_tensor

    child.load_state_dict(child_state)
    child.input_size = input_size
    child.output_size = output_size
    #print(f"[Controller Crossover] Parent1: {brain_signature(parent1)} | Parent2: {brain_signature(parent2)} | Child: {brain_signature(child)}")
    return child

def brain_signature(brain):
    return tuple(torch.cat([p.flatten() for p in brain.parameters()]).detach().numpy().round(decimals=4))

# COEVOLUTION ALGORITHM
def run_coevolution(scenario, seed):
    # Initialize populations with paired, compatible structures and controllers
    structure_population = []
    brain_population = []

    while len(structure_population) < POP_SIZE:
        robot = create_random_robot()
        if np.any(robot > 0) and is_connected(robot):
            try:
                controller, _, _, _ = create_random_controller(robot, scenario)
                structure_population.append(robot)
                brain_population.append(controller)
            except:
                continue

    best_fitness = -float('inf')
    best_pair = None
    fitness_history = []
    generation_best_history = []
    no_improvement = 0
    stagnation_point = 0
    phase_cycle = 5
    current_phase = "controller" 
    generation = 0
    while generation < GENERATIONS:
        t_start_gen = time.time()
        # Evaluate all pairs for fitness matrix and extract bests
        t_eval_start = time.time()
        fitness_matrix = np.full((len(structure_population), len(brain_population)), -5.0)
        for i, s in enumerate(structure_population):
            for j, b in enumerate(brain_population):
                # Compatibility checked inside evaluate_fitness
                fitness_matrix[i, j] = evaluate_fitness(s, b, scenario)
        print(f"[{scenario}][Seed {seed}] Gen {generation}: Fitness evaluation took {time.time() - t_eval_start:.2f} seconds")
        # Find best structure (across all brains) and best brain (across all structures)
        struct_fitness = fitness_matrix.max(axis=1)
        ctrl_fitness = fitness_matrix.max(axis=0)
        best_struct_idx = int(np.argmax(struct_fitness))
        best_ctrl_idx = int(np.argmax(ctrl_fitness))
        best_structure = structure_population[best_struct_idx]
        best_brain = brain_population[best_ctrl_idx]
        # The best pair is the one with highest fitness in the matrix
        best_indices = np.unravel_index(np.argmax(fitness_matrix, axis=None), fitness_matrix.shape)
        best_pair_struct = structure_population[best_indices[0]]
        best_pair_brain = brain_population[best_indices[1]]
        best_pair_fitness = fitness_matrix[best_indices]
        generation_best = best_pair_fitness
        generation_best_history.append(generation_best)
        fitness_history.append(max(generation_best, fitness_history[-1] if fitness_history else -float("inf")))

        n_elite = max(1, int(POP_SIZE * ELISTIM))
        elite_structures = [structure_population[i] for i in np.argsort(struct_fitness)[-n_elite:]] if len(structure_population) >= n_elite else []
        elite_brains = [brain_population[i] for i in np.argsort(ctrl_fitness)[-n_elite:]] if len(brain_population) >= n_elite else []

        if best_pair_fitness > best_fitness:
            best_pair = (copy.deepcopy(best_pair_struct), copy.deepcopy(best_pair_brain))
            best_fitness = best_pair_fitness
            stagnation_point = generation
            no_improvement = 0
        else:
            no_improvement += 1

        t_evolution_start = time.time()

        if current_phase == "structure":
            # Evolve structures, keep brains fixed
            new_structures = []
            attempts = 0
            while len(new_structures) < POP_SIZE and attempts < POP_SIZE * 10:
                p1 = tournament_selection_from_list(structure_population, struct_fitness)
                p2 = tournament_selection_from_list(structure_population, struct_fitness)
                child = crossover_structure(p1, p2)
                child = mutate_structure(child)
                if np.any(child > 0) and is_connected(child):
                    new_structures.append(child)
                attempts += 1
            # Remove duplicates
            flat_structs = np.array([ind.flatten() for ind in new_structures])
            _, unique_idx = np.unique(flat_structs, axis=0, return_index=True)
            temp_structures = [new_structures[i] for i in sorted(unique_idx)]
            # Ensure minimum population size
            while len(temp_structures) < POP_SIZE:
                robot = create_random_robot()
                if np.any(robot > 0) and is_connected(robot):
                    temp_structures.append(robot)
            # Adapt best controller to each structure 
            new_brains = []
            for s in temp_structures:
                adapted = adapt_controller_to_structure(best_brain, s, scenario)
                new_brains.append(adapted)
            # Elitism for structures, brains remain untouched
            structure_population = elite_structures + temp_structures[:-n_elite]
        else:
            # Evolve controllers, keep structures fixed
            new_brains = []
            attempts = 0
            while len(new_brains) < POP_SIZE and attempts < POP_SIZE * 10:
                p1 = tournament_selection_from_list(brain_population, ctrl_fitness)
                p2 = tournament_selection_from_list(brain_population, ctrl_fitness)
                child = crossover_controller(p1, p2)
                child = mutate_controller(child)
                # Adapt controller to best structure if needed
                adapted = adapt_controller_to_structure(child, best_structure, scenario)
                # Only keep if compatible
                if hasattr(adapted, "input_size") and hasattr(adapted, "output_size"):
                    try:
                        expected_input, expected_output = get_input_output_size(best_structure, scenario)
                        if adapted.input_size == expected_input and adapted.output_size == expected_output:
                            new_brains.append(adapted)
                    except:
                        continue
                attempts += 1
            # Remove duplicates
            seen_brains = set()
            filtered_brains = []
            for b in new_brains:
                sig = brain_signature(b)
                if sig not in seen_brains:
                    seen_brains.add(sig)
                    filtered_brains.append(b)
            while len(filtered_brains) < POP_SIZE:
                # Fill up with random controllers adapted to best structure
                try:
                    controller, _, _, _ = create_random_controller(best_structure, scenario)
                    filtered_brains.append(controller)
                except:
                    continue
            # Elitism for brains, structures remain untouched
            brain_population = elite_brains + filtered_brains[:-n_elite]
        print(f"[{scenario}][Seed {seed}] Gen {generation}: Population evolution took {time.time() - t_evolution_start:.2f} seconds")

        struct_signatures = set(str(s.flatten()) for s in structure_population)
        brain_signatures = set(brain_signature(b) for b in brain_population)
        print(f"[{scenario}][Seed {seed}] Gen {generation}: Unique Structures = {len(struct_signatures)} | Unique Controllers = {len(brain_signatures)}")
        print(f"[{scenario}][Seed {seed}] Gen {generation}: Generation Best = {generation_best:.4f} | Cumulative Best = {fitness_history[-1]:.4f}")
        print(f"[{scenario}][Seed {seed}] Gen {generation}: Best Structure Fitness = {max(struct_fitness):.4f}, Best Controller Fitness = {max(ctrl_fitness):.4f}")
        print(f"[{scenario}][Seed {seed}] Gen {generation}: Time = {time.time() - t_start_gen:.2f} seconds")

        # Phase switching
        if (generation + 1) % phase_cycle == 0:
            current_phase = "structure" if current_phase == "controller" else "controller"
            print(f"[{scenario}][Seed {seed}] Switching to phase: {current_phase}")
        generation += 1

    best_structure, best_controller = best_pair
    return best_structure, best_controller, fitness_history, generation_best_history, stagnation_point

# PHASE 1: Run CoEvo
def run_phase1():
    all_records = []

    for scenario in SCENARIOS:
        fitness_per_seed = []

        for seed in SEEDS:
            print(f"\n{scenario} - CoEvo - Seed {seed}")
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            best_structure, best_controller, fitness_history, generation_best_history, stagnation_point = run_coevolution(scenario, seed)

            # Save fitness history
            csv_seed_path = os.path.join("csvs", "fase3", "phase1", scenario, f"seed_{seed}")
            os.makedirs(csv_seed_path, exist_ok=True)
            df_fitness = pd.DataFrame({
                'generation': list(range(GENERATIONS)),
                'generation_best': generation_best_history,
                'best_fitness_cumulative': fitness_history
            })
            df_fitness.to_csv(os.path.join(csv_seed_path, 'fitness.csv'), index=False)

            # Save structure
            structure_path = os.path.join("structures", "fase3", "phase1", scenario, f"seed_{seed}")
            os.makedirs(structure_path, exist_ok=True)
            np.save(os.path.join(structure_path, 'best_structure.npy'), best_structure)

            # Save controller
            model_path = os.path.join("models", "fase3", "phase1", scenario, f"seed_{seed}")
            os.makedirs(model_path, exist_ok=True)
            torch.save(best_controller.state_dict(), os.path.join(model_path, 'best_controller.pt'))

            # Save bundle (controler + structure)
            bundle_path = os.path.join("bundles", "fase3", "phase1", scenario, f"seed_{seed}")
            os.makedirs(bundle_path, exist_ok=True)
            np.savez(
                os.path.join(bundle_path, "bundle.npz"),
                structure=best_structure,
                controller_path=os.path.join(model_path, 'best_controller.pt'),
                input_size=best_controller.input_size,
                output_size=best_controller.output_size
            )

            fitness_per_seed.append(fitness_history)

            final_fitness = fitness_history[-1]
            all_records.append({"Scenario": scenario, "Algorithm": "CoEvo", "Seed": seed, "FinalFitness": final_fitness, "StagnationGen": stagnation_point})

    summary_folder = os.path.join("csvs", "fase3", "phase1")
    os.makedirs(summary_folder, exist_ok=True)
    df_summary = pd.DataFrame(all_records)
    df_summary.to_csv(os.path.join(summary_folder, "summary_phase1.csv"), index=False)

def plot_phase1_mean_curves():
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = itertools.cycle(['blue', 'green', 'orange', 'purple', 'red'])

    for scenario in SCENARIOS:
        folder = os.path.join("csvs", "fase3", "phase1", scenario)
        all_runs = []
        for seed in SEEDS:
            path = os.path.join(folder, f"seed_{seed}", "fitness.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                all_runs.append(df["best_fitness_cumulative"].values)

        if all_runs:
            df_all = pd.DataFrame(all_runs).T
            mean_fitness = df_all.mean(axis=1)
            std_fitness = df_all.std(axis=1)
            color = next(colors)
            ax.plot(range(1, len(mean_fitness)+1), mean_fitness, label=scenario, color=color)

    ax.set_title("Fase 3 - Phase 1 Mean Fitness Evolution (Best Fitness Cumulative)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    plot_folder = os.path.join("plots", "fase3", "phase1", "mean_curves")
    os.makedirs(plot_folder, exist_ok=True)
    i = 1
    path = lambda i: os.path.join(plot_folder, f"mean_curve_phase1_{i}.png")
    while os.path.exists(path(i)):
        i += 1
    plt.savefig(path(i))
    plt.close()

def plot_phase1_boxplots():
    df_summary = pd.read_csv("csvs/fase3/phase1/summary_phase1.csv")
    fig, ax = plt.subplots()
    all_data = []
    labels = []

    for scenario in df_summary["Scenario"].unique():
        subset = df_summary[df_summary["Scenario"] == scenario]
        values = subset["FinalFitness"].values
        all_data.append(values)
        labels.append(scenario)

    box = ax.boxplot(all_data, patch_artist=True, tick_labels=labels)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'violet']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    means = [np.mean(values) for values in all_data]
    for i, mean in enumerate(means):
        ax.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=9)

    ax.set_title("Fase 3 - Phase 1: Algorithm Comparison")
    ax.set_ylabel("Final Fitness")
    ax.grid(True)
    plt.tight_layout()

    folder = os.path.join("plots", "fase3", "phase1", "boxplots")
    os.makedirs(folder, exist_ok=True)
    i = 1
    path = lambda i: os.path.join(folder, f"boxplot_phase1_{i}.png")
    while os.path.exists(path(i)):
        i += 1
    plt.savefig(path(i))
    plt.close()

def load_bundle_and_visualize_gif_phase1():
    import glob

    base_path = os.path.join("bundles", "fase3", "phase1")
    for scenario in SCENARIOS:
        bundle_paths = glob.glob(os.path.join(base_path, scenario, "seed_*", "bundle.npz"))
        for bundle_path in bundle_paths:
            print(f"Visualizing: {bundle_path}")
            data = np.load(bundle_path, allow_pickle=True)
            structure = data["structure"]
            controller_path = data["controller_path"].item()

            env, sim, input_size, output_size, _ = init_env(structure, scenario)
            brain = NeuralController(input_size, output_size)
            brain.load_state_dict(torch.load(controller_path, weights_only=True))
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')
            state = env.reset()[0]
            for t in range(STEPS):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = brain(state_tensor).detach().numpy().flatten()
                viewer.render('screen')
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            viewer.close()
            env.close()

# PHASE 2: Hyperparameter Tuning
def run_phase2_hyperparameter_tuning():
    """
    Runs hyperparameter tuning for CoEvolution by varying MUTATION_RATE and CROSSOVER_RATE. Collects final fitness for each run and performs statistical tests.
    Saves results as CSV.
    """
    parameter_grid = {
        'MUTATION_RATE': [0.1, 0.2, 0.4],
        'CROSSOVER_RATE': [0.6, 0.8, 1.0],
        'ELITISM_RATE': [0.1]
    }
    seeds_subset = SEEDS 
    scenarios = SCENARIOS
    results = []
    fitness_records = []
    param_combinations = list(itertools.product(
        parameter_grid['MUTATION_RATE'],
        parameter_grid['CROSSOVER_RATE'],
        parameter_grid['ELITISM_RATE']
    ))
    for mut_rate, cross_rate, elit_rate in param_combinations:
        for scenario in scenarios:
            for seed in seeds_subset:
                global MUTATION_RATE, CROSSOVER_RATE, ELISTIM
                MUTATION_RATE = mut_rate
                CROSSOVER_RATE = cross_rate
                ELISTIM = elit_rate
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                best_structure, best_controller, fitness_history, generation_best_history, stagnation_point = run_coevolution(scenario, seed)
                final_fitness = fitness_history[-1]
                results.append({
                    'MUTATION_RATE': mut_rate,
                    'CROSSOVER_RATE': cross_rate,
                    'ELITISM_RATE': elit_rate,
                    'Scenario': scenario,
                    'Seed': seed,
                    'FinalFitness': final_fitness
                })
                fitness_records.append({
                    'param_tuple': (mut_rate, cross_rate, elit_rate),
                    'fitness': final_fitness
                })
                print(f"[Hyperparam] MUT={mut_rate} CROSS={cross_rate} ELIT={elit_rate} SCENARIO={scenario} SEED={seed} => FinalFitness={final_fitness:.4f}")

                csv_seed_path = os.path.join("csvs", "fase3", "phase2")
                os.makedirs(csv_seed_path, exist_ok=True)
                df_fitness = pd.DataFrame({
                    'generation': list(range(len(fitness_history))),
                    'generation_best': generation_best_history,
                    'best_fitness_cumulative': fitness_history
                })
                filename = f"fitness_M{mut_rate}_C{cross_rate}_E{elit_rate}_S{seed}_T{scenario}.csv"
                df_fitness.to_csv(os.path.join(csv_seed_path, filename), index=False)

    df_results = pd.DataFrame(results)
    os.makedirs("csvs/fase3/phase2", exist_ok=True)
    csv_path = "csvs/fase3/phase2/hyperparameter_tuning.csv"
    df_results.to_csv(csv_path, index=False)

def plot_phase2_mean_curves():
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = itertools.cycle(['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink', 'cyan'])

    tuning_folder = "csvs/fase3/phase2"
    all_files = glob.glob(os.path.join(tuning_folder, "fitness_M*_C*_E*_S*_T*.csv"))

    param_runs = {}
    for file in all_files:
        base = os.path.basename(file)
        key = "_".join(base.split("_")[:6]) 
        if key not in param_runs:
            param_runs[key] = []
        param_runs[key].append(file)

    for key, files in param_runs.items():
        all_runs = []
        for path in files:
            df = pd.read_csv(path)
            all_runs.append(df["best_fitness_cumulative"].values)

        if all_runs:
            df_all = pd.DataFrame(all_runs).T
            mean_fitness = df_all.mean(axis=1)
            std_fitness = df_all.std(axis=1)
            color = next(colors)
            ax.plot(range(1, len(mean_fitness)+1), mean_fitness, label=key, color=color)

    ax.set_title("Fase 3 - Phase 2 Mean Fitness Evolution (Best Fitness Cumulative)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    ax.legend(fontsize='small', loc='upper left')
    ax.grid(True)
    plt.tight_layout()

    plot_folder = os.path.join("plots", "fase3", "phase2", "mean_curves")
    os.makedirs(plot_folder, exist_ok=True)
    i = 1
    path = lambda i: os.path.join(plot_folder, f"mean_curve_phase2_{i}.png")
    while os.path.exists(path(i)):
        i += 1
    plt.savefig(path(i))
    plt.close()

def plot_phase2_boxplots():
    df_summary = pd.read_csv("csvs/fase3/phase2/hyperparameter_tuning.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    all_data = []
    labels = []

    df_summary["combo"] = df_summary.apply(lambda row: f"M{row['MUTATION_RATE']}_C{row['CROSSOVER_RATE']}_E{row['ELITISM_RATE']}", axis=1)

    for combo in sorted(df_summary["combo"].unique()):
        subset = df_summary[df_summary["combo"] == combo]
        values = subset["FinalFitness"].values
        all_data.append(values)
        labels.append(combo)

    box = ax.boxplot(all_data, patch_artist=True, labels=labels)
    colors = itertools.cycle(['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'violet', 'lightgray', 'lightpink'])
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    means = [np.mean(values) for values in all_data]
    for i, mean in enumerate(means):
        ax.text(i + 1, mean + 0.05, f"{mean:.2f}", ha='center', fontsize=8)

    ax.set_title("Fase 3 - Phase 2: Hyperparameter Configurations Comparison")
    ax.set_ylabel("Final Fitness")
    ax.set_xlabel("Hyperparameter Combination")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    plt.tight_layout()

    folder = os.path.join("plots", "fase3", "phase2", "boxplots")
    os.makedirs(folder, exist_ok=True)
    i = 1
    path = lambda i: os.path.join(folder, f"boxplot_phase2_{i}.png")
    while os.path.exists(path(i)):
        i += 1
    plt.savefig(path(i))
    plt.close()

# PHASE 3: Statistical Comparison
def run_phase3_statistical_analysis_phase2():
    """
    Perform statistical comparison of hyperparameter combinations from Phase 2.
    Uses final fitness from 'hyperparameter_tuning.csv'.
    Tests normality and variance homogeneity, then applies ANOVA or Kruskal-Wallis.
    Saves results in CSV.
    """
    df = pd.read_csv("csvs/fase3/phase2/hyperparameter_tuning.csv")
    os.makedirs("csvs/fase3/phase3", exist_ok=True)

    df["combo"] = df.apply(lambda row: f"GA_MUT{row['MUTATION_RATE']}_CROSS{row['CROSSOVER_RATE']}_ELIT{row['ELITISM_RATE']}", axis=1)
    grouped = df.groupby("combo")["FinalFitness"].apply(list).to_dict()

    all_data = [v for v in grouped.values()]
    normal = True
    for group_data in all_data:
        if len(group_data) < 3:
            normal = False
            break
        stat, p = shapiro(group_data)
        if p < 0.05:
            normal = False
            break

    if normal:
        test_used = "ANOVA"
        stat, p_value = f_oneway(*all_data)
    else:
        test_used = "Kruskal-Wallis"
        stat, p_value = kruskal(*all_data)

    print(f"\nPhase 3 Analysis:")
    print(f"Test Used: {test_used}")
    print(f"Statistic={stat:.4f}, p-value={p_value:.4f}")

    significance = "Yes" if p_value < 0.05 else "No"

    df_result = pd.DataFrame({
        "Test Used": [test_used],
        "Statistic": [stat],
        "p-value": [p_value],
        "Significant": [significance]
    })
    df_result.to_csv("csvs/fase3/phase3/statistical_test_phase2.csv", index=False)

    mean_fitness = df.groupby("combo")["FinalFitness"].mean()
    best_combo = mean_fitness.idxmax()
    best_parts = best_combo.split("_")
    best_data = {
        "best_mutation": [best_parts[1].replace("MUT", "")],
        "best_crossover": [best_parts[2].replace("CROSS", "")],
        "best_elitism": [best_parts[3].replace("ELIT", "")]
    }
    df_best = pd.DataFrame(best_data)
    df_best.to_csv("csvs/fase3/phase3/best_ga_phase2.csv", index=False)

# Main Function
def main():
    print("Running Phase 1: CoEvolution")
    run_phase1()
    plot_phase1_mean_curves()
    plot_phase1_boxplots()
    load_bundle_and_visualize_gif_phase1()

    print("Running Phase 2: Hyperparameter Tuning")
    run_phase2_hyperparameter_tuning()
    plot_phase2_mean_curves()
    plot_phase2_boxplots()

    print("Running Phase 3: Statistical Analysis")
    run_phase3_statistical_analysis_phase2()

if __name__ == "__main__":
    main()
