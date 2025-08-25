# autossm-2afc-thesis
Automated discovery of sequential sampling models for 2-alternative forced-choice tasks (thesis version)

This project contains the code and data for the thesis "Automated discovery of sequential sampling models for
2-alternative forced-choice tasks" by Ibrahim Muhip Tezcan, supervised by Daniel Weinhardt and Prof. Dr. Sebastian Musslick.

## Running the code

Before running the code, make sure to install the required packages. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

Moreover, PyBeam needs to be installed via the provided package under the ddm folder:

```bash
cd ddm/pybeam_simulate
pip install -e .
```

Finally, the custom PyBeam model `get_traces` need to be installed as well:

```bash
cd ddm/custom_models/get_traces
python setup.py build_ext --inplace
```

Once the required packages are installed, you can run the code by executing the following command in your terminal:

```bash
python rnn_main.py
```

Make sure to run this command in the root directory of the project, where the `rnn_main.py` file is located.
This will start the training process using the configuration specified in `simpleddm.yml` by default, training it once.

You can specify a different configuration file by using the `--config` argument:

```bash
python rnn_main.py --config <path_to_config_file>
```

Currently two more configurations are provided in the `configs` folder: increasing_drift.yml and ugm.yml, preconfigured
for the given models. Running the linearly changing drifts model also requires installing the `changing_drift` PyBeam
model under `ddm/custom_models/changing_drift`:

```bash
cd ddm/custom_models/changing_drift
python setup.py build_ext --inplace
```

## Multiple trainings
To run training multiple times, you can specify the number of training runs by using the `--train_count` argument.

Example:
```bash
python rnn_main.py --train_count 10
```

To run training with multiple drift rates, you can specify the number of drift rates by using the `--drift_count` argument.

Example:
```bash
python rnn_main.py --drift_rates 0.1 0.5 1.0 2.0 4.0
```

## Running on the GPU

Currently all configurations are set to run on cpu by default. You can change it to "cuda" or "mps" (for Apple silicon)
in the configuration file (training_params -> device).

## Results

Results can be found in analysis.ipynb. To run the notebook, you need to have the trained models. You can either ask me
or train the models yourself and change the notebook accordingly to load the newly trained models.

## Troubleshooting

For python 3.11 - if you get “no module named Cython” error after installing Cython, try pip install wheel --upgrade