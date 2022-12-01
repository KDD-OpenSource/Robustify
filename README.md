This repository contains the code and experimental configuration files for the ICDM2022 Paper [Post-Robustifying Deep Anomaly Detection Ensembles by Model Selection](https://ls9-www.cs.tu-dortmund.de/publications/ICDM2022.pdf)

"Model_training" contains the code for generating the anomaly detectors used for this Paper.
Also take a look at DEAN here: https://github.com/psorus/DEAN

"Model_verification" contains the code for verifying the given models.
Note that in order to verify, one must have [Marabou](https://github.com/NeuralNetworkVerification/Marabou) installed.

To verify a given model, first save in it e.g. './models/trained_models/deepsvdd_cardio/models' as demonstrated by the example. Thereafter you can run 
```bash
python3 main.py configs/reprod/icdm/config_test_svdd_cardio.yaml
```
and find the results in './reports'. If you choose to save the model in a different folder you will need to adjust the config file accordingly.

## Authors/Contributors
* [Benedikt Böing](https://github.com/bboeing)
* [Simon Klüttermann](https://github.com/psorus)
* [Emmanuel Müller](https://github.com/emmanuel-mueller)
