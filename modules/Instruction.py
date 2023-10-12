import numpy as np
import random
from modules.Dataset import Dataset
from modules.Parameter import Parameter

class Instruction:
    def __init__(self):
        self.source_select = random.randint(0, Parameter.source_select - 1)
        self.target_index = random.randint(0, Parameter.target_index - 1)
        self.operator_select = random.randint(0, Parameter.operator_select - 1)
        self.source_index = random.randint(0, Parameter.source_index - 1)

    def compute_instruction_vectorized(self, row_data, registers):
        source_value = np.zeros(row_data.shape[0])
        
        if self.source_select == 0:
            source_value = row_data[:, self.source_index % row_data.shape[1]]
        else:
            source_value = registers[:, self.source_index % Parameter.register_count]

        self.apply_operator_vectorized(self.target_index, self.operator_select, source_value, registers)

    def apply_operator_vectorized(self, target_index, operator, source_value, registers):
        target = target_index % Parameter.register_count

        if operator == 0:
            registers[:, target] += source_value
        elif operator == 1:
            registers[:, target] -= source_value
        elif operator == 2:
            registers[:, target] *= 2  # Multiplies by 2
        elif operator == 3:
            registers[:, target] /= 2  # Divides by 2
