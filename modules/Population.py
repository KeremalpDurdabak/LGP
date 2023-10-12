import copy
import random
from modules.Dataset import Dataset
from modules.Individual import Individual  # Corrected the typo here
from modules.Parameter import Parameter

class Population:
    def __init__(self):
        self.individuals = self.create_individuals()

    def create_individuals(self):
        return [Individual() for _ in range(Parameter.pop_count)]

    def compute_fitness(self):
        for individual in self.individuals:
            individual.compute_fitness()
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def generate_next_gen(self):
        num_to_remove = int(Parameter.gap_percentage * Parameter.pop_count)

        # Remove the worst individuals
        self.individuals = self.individuals[:-num_to_remove]

        # Generate children through crossover and mutation
        children = self.crossover_and_mutate(num_to_remove)

        # Compute fitness for the new children
        for child in children:
            child.compute_fitness()

        # Replace the worst individuals with the new children
        self.individuals.extend(children)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def crossover_and_mutate(self, num_children):
        children = []
        
        while len(children) < num_children:
            # Select two random parents
            parent1 = random.choice(self.individuals)
            parent2 = random.choice(self.individuals)
            
            child1, child2 = self.perform_crossover(parent1, parent2)
            
            # Add to children list
            children.extend([child1, child2])
        
        # Truncate extra children if any
        children = children[:num_children]
        
        # Apply mutation
        for i in range(len(children)):
            if random.random() < Parameter.mutation_prob:
                self.perform_mutation(children[i])
        
        return children


    def perform_crossover(self, parent1, parent2):
        # Initialize new individuals for children
        child1 = Individual()
        child2 = Individual()

        # Find the minimum instruction count between the two parents
        min_instruction_count = min(len(parent1.instructions), len(parent2.instructions))

        # Single-point or double-point crossover (50% chance for each)
        if random.random() < 0.5:
            # Single-point crossover
            crossover_point = random.randint(1, min_instruction_count - 1)
            child1.instructions = copy.deepcopy(parent1.instructions[:crossover_point]) + copy.deepcopy(parent2.instructions[crossover_point:])
            child2.instructions = copy.deepcopy(parent2.instructions[:crossover_point]) + copy.deepcopy(parent1.instructions[crossover_point:])
        else:
            # Double-point crossover
            point1 = random.randint(1, min_instruction_count - 1)
            point2 = random.randint(point1, min_instruction_count - 1)
            child1.instructions = copy.deepcopy(parent1.instructions[:point1]) + copy.deepcopy(parent2.instructions[point1:point2]) + copy.deepcopy(parent1.instructions[point2:])
            child2.instructions = copy.deepcopy(parent2.instructions[:point1]) + copy.deepcopy(parent1.instructions[point1:point2]) + copy.deepcopy(parent2.instructions[point2:])

        return child1, child2


    def perform_mutation(self, individual):
        for instruction in individual.instructions:
            # 50% chance for each instruction to get one of their bits changed
            if random.random() < 0.5:
                # Decide which attribute to mutate
                attribute_to_mutate = random.choice(['source_select', 'target_index', 'operator_select', 'source_index'])
                
                # Mutate the chosen attribute based on its valid range
                if attribute_to_mutate == 'source_select':
                    instruction.source_select = random.randint(0, Parameter.source_select - 1)
                elif attribute_to_mutate == 'target_index':
                    instruction.target_index = random.randint(0, Parameter.target_index - 1)
                elif attribute_to_mutate == 'operator_select':
                    instruction.operator_select = random.randint(0, Parameter.operator_select - 1)
                elif attribute_to_mutate == 'source_index':
                    instruction.source_index = random.randint(0, Parameter.source_index - 1)
