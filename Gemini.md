this project want to reproduce pangu drug model from the paper "PanGu Drug Model: Learn a Molecule Like a Human" using coding agent.
please work on the following tasks:
1. read the paper 
2. work as a deeplearning architect to design the neaural network architecture according to the paper as model v1
4. implement the model using pytorch, design the code to be modular so that new archtectures could be adopted easily with the same settings, use uv to manage the packages
5. Download and process the ZINC dataset for training and testing. This involves the following steps:
    - **Data Source**: Identify and use a suitable subset from the ZINC database. We will start with a smaller subset in SMILES format for easier initial development and testing.
    - **Download**: Use a command-line tool like `wget` or `curl` within the `bootstrap.sh` script to download the chosen dataset. This ensures the process is automated and reproducible.
    - **Processing Library**: Utilize the `RDKit` library, a powerful tool for cheminformatics in Python, to process the raw SMILES data. This will be added as a project dependency and managed with `uv`.
    - **Data Transformation**: The `src/data_loader.py` script will be responsible for:
        - Reading the downloaded SMILES file.
        - Converting each SMILES string into a molecular graph representation using `RDKit`.
        - Extracting relevant features for the model (e.g., atom types, bond types).
        - Saving the processed data into a clean, ready-to-use format (e.g., PyTorch `.pt` files) in a `data/processed` directory.
6. use 'bootstrap.sh' to manage the project, including the  virtual env setup, data download, train and test