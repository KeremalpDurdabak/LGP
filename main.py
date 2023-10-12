from modules.Parameter import Parameter
from modules.Dataset import Dataset
from modules.Population import Population
from modules.Display import Display

def main():
    # Initialize and Set the Dataset
    Dataset.load_data(Parameter.data_path)
    Dataset.resample_data(200,'stratified') # 'uniform' or 'stratified'

    # Initialize and Generate the Initial Population
    population = Population()
    population.compute_fitness()

    for generation in range(1, Parameter.generations + 1):
        # Use Display method to report best individual's fitness
        Display.report_best_individual(population, generation)
        
        # Generate the next generation
        population.generate_next_gen()
        if generation % 10 == 0:
            Dataset.resample_data(200,'stratified') # 'uniform' or 'stratified'

    # #Compute best individual's score on the test dataset
    # population.compute_best_individual_test_fitness()
    
    # # Display overall metrics as graphs    
    # Display.report_overall_performance()

if __name__ == "__main__":
    main()
