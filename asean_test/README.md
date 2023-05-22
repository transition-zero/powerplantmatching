Once you have installed anaconda or mamba you can run the following command: 

```bash
mamba env create -f requirements.yaml
```

This will create a new conda environment called `powerplantmatching-feo` with all the required packages.

## Usage

You can then run the following command to activate the environment:

```bash
mamba activate powerplantmatching-feo
```

Finally, you can run the test.py file with the following commnad:

```bash 
python asean_test/test.py
```

If you want to change the way powerplantmatching behaves you can do so in the 'asean_test/config.yaml' file. 