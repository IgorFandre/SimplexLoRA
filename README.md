## Requirements 
    python 3.11.7

## Dependencies installation
Please follow these steps (required!):

0. Install python3.11.7
    - Install conda
    - Run:
      ```bash
      conda create --name python3.11.7 python=3.11.7
      conda activate python3.11.7
      ```

1. Create virtual env
```bash
python -m venv venv
source venv/bin/activate
```

2. Update pip
```bash
pip install pip==24.1.1
```

3. Install rest of packages from requirements.txt
```bash
pip install -r requirements.txt
```

4. Install local version of peft
```bash
pip install ./peft
```

#

For the further runs use:
```bash
conda activate python3.11.7
source venv/bin/activate
```

## Run experiments
To run our experiments, use run_glue_experiments.sh

```
./run_glue_experiments.sh
```

To conduct your own experiments, check [glue_experiment/scripts/](./glue_experiment/scripts/)
