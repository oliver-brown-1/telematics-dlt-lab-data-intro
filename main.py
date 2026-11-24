from insurance_telematics import TripSimulator, TelematicsScoringPipeline

def main():
    sim = TripSimulator(seed=42)
    trips_df, claims_df = sim.simulate(n_drivers=100, trips_per_driver=50)

    pipe = TelematicsScoringPipeline(n_hmm_states=3)
    pipe.fit(trips_df, claims_df)
    predictions = pipe.predict(trips_df)

    print("Predictions:")
    print(predictions.head())


if __name__ == "__main__":
    main()
