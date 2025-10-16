# Survival of the Softest: Evolving Adaptive Soft Robots  

This repository contains the **code, report, and experiments** for the project **“Survival of the Softest”**, developed for the **Evolutionary Computation** course at the **University of Coimbra (2024/2025)**.  

## Project Overview  
The goal of this project is to **evolve adaptive soft robots** using **evolutionary computation techniques** within the **EvoGym simulation environment**.  
The project explores three progressive stages:  
1. **Evolving the robot structure** with a fixed controller.  
2. **Evolving the controller** with a fixed robot morphology.  
3. **Co-evolving both structure and controller** simultaneously to achieve adaptive locomotion.  

Different **bio-inspired algorithms** were implemented and compared, including **Genetic Algorithms (GA)**, **Evolutionary Strategies (ES)**, and **Differential Evolution (DE)**, focusing on how evolutionary processes can optimize both morphology and control in soft robots.  

## Available Files  
- **`random_structure.py`** – Evolves robot morphologies using a Genetic Algorithm.  
- **`random_controler.py`** – Evolves neural network controllers for predefined robot structures.  
- **`random_structure_and_controller.py`** – Implements the co-evolutionary algorithm for simultaneous evolution of structure and control.  
- **`FinalReport-2024154464.pdf`** – Comprehensive report describing methodology, algorithms, experiments, and results.  

**The EvoGym environment and datasets are NOT included in this repository** due to size and licensing restrictions.  

## Technologies Used  
- **Python**  
- **NumPy, Matplotlib**  
- **EvoGym** (MIT CSAIL simulation platform)  
- **Evolutionary Algorithms:** Genetic Algorithm (GA), Evolutionary Strategies (ES), Differential Evolution (DE)  

## Contact  
For any questions or collaborations, feel free to reach out:  
**Email:** martinusboom@gmail.com  
**GitHub:** [MartinusBoomUc2024154464](https://github.com/MartinusBoomUc2024154464)
