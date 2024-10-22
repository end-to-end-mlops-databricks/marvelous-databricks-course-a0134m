import pandas as pd
import yaml

class DataProcessor:
    def __init__(self, filepath, config):
        self.df = self.load_data(filepath)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def preprocess_data(self):
        # Remove rows with missing target
        target = self.config['target']
        self.df = self.df.dropna(subset=[target])