import pandas as pd


class DatasetKnowledge():
    def __init__(self, csv_file_path):
        super().__init__()

        self.file_path = csv_file_path
        self.data_original = pd.read_csv(self.file_path)
        self.X = self.data_original
        self.Y = pd.DataFrame()
        self.columns = list(self.data_original.columns)

        # column functions
        self.targets = []
        self.features = self.columns
        self.numerical_features = []
        self.numerical_features_circular = []
        self.categorical_features = []
        self.categorical_features_nominal = []
        self.categorical_features_ordinal = []
        self.features_text = []
        self.features_removed = []

    def show_features(self):
        pass

    def define_targets(self, targets_list):
        for target in targets_list:
            self.targets.append(target)
            try:
                self.features.remove(target)
                self.numerical_features.remove(target)
                self.numerical_features_circular.remove(target)
                self.categorical_features.remove(target)
                self.categorical_features_nominal.remove(target)
                self.categorical_features_ordinal.remove(target)
                self.features_text.remove(target)
            except:
                pass



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = DatasetKnowledge(csv_file_path='train.csv')
    print(dataset.X.head(10))
    print(dataset.features)
    dataset.define_targets(['SalePrice'])
    print(dataset.features)

    print(dataset.targets)


