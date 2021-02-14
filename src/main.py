from Simulation import game

if __name__ == "__main__":
    game(
        population_size=300,
        n_best=90,
        epochs=4000,
        turns_per_epoch=50,
        n_word=2,
        turns_to_sleep=10,
        verbose=True
    )
