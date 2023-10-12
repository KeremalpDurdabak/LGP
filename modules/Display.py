class Display:
    @staticmethod
    def report_best_individual(population, generation):
        best_fitness = population.individuals[0].fitness
        print(f"Generation: {generation}, Best individual's fitness: {best_fitness:.2f}")

        # Collect and sort every individual's fitness
        all_fitness = [f"{individual.fitness:.2f}" for individual in population.individuals]
        all_fitness.sort(reverse=True)  # Assuming you want to sort in descending order

        # Report sorted fitness scores
        print(f"All Fitness Scores (Sorted): {all_fitness}")
