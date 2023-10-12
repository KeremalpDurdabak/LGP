import random

import numpy as np
from modules.Dataset import Dataset
from modules.Instruction import Instruction

from modules.Parameter import Parameter


class Individual:
    def __init__(self):
        self.instructions = self.create_instructions()
        self.fitness = 0

    def create_instructions(self):
        num_instructions = random.randint(2, Parameter.max_instruction - 1)
        return [Instruction() for _ in range(num_instructions)]

    def compute_fitness(self):
        row_data = Dataset.X_train.values  # Assuming X_train is a DataFrame
        num_rows = len(Dataset.X_train)
        registers = np.zeros((num_rows, Parameter.register_count))

        for instruction in self.instructions:
            instruction.compute_instruction_vectorized(row_data, registers)

        if Dataset.problem_type == 'Regression':
            y_train_values = Dataset.y_train
            mse_accumulator = np.mean((registers[:, 0] - y_train_values) ** 2)
            self.fitness = -mse_accumulator  # Negative because we usually try to maximize fitness

        elif Dataset.problem_type == 'Classification':
            y_train_values = Dataset.y_train  # Assuming it's already a NumPy array
            predicted_labels = np.argmax(registers[:, :Dataset.y_train.shape[1]], axis=1)
            true_labels = np.argmax(y_train_values, axis=1)
            correct_count = np.sum(predicted_labels == true_labels)
            self.fitness = correct_count / num_rows

