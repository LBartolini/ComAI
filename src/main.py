from Simulation import game

if __name__ == "__main__":
    game(
        population_size=900,
        n_best=250,
        epochs=25,
        turns_per_epoch=150,
        n_word=2,
        turns_to_sleep=100,
        verbose=True
    )
