import pandas as pd


class DatasetKnowledge():
    def __init__(self, csv_file_path):
        super().__init__()

        self.file_path = csv_file_path
        self.data_original = pd.read_csv(self.file_path)
        self.X = self.data_original
        self.Y = pd.DataFrame()
        self.columns = self.data_original.columns.to_list()
        self.columns_number=len(self.columns)

        # column functions
        self.targets = []
        self.targets_number = 0

        self.features = self.columns
        self.features_number=len(self.features)

        #Numerical features
        self.numerical_features = list(self.X._get_numeric_data().columns)
        self.numerical_features_number=len(self.numerical_features)
        self.numerical_features_circular = []
        self.numerical_features_circular_number=0

        #Categorical features
        self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        self.categorical_features_number=len(self.categorical_features)
        self.categorical_features_nominal = self.categorical_features #in the first step we assume that data are nominal
        self.categorical_features_nominal_number=len(self.categorical_features_nominal)
        self.categorical_features_ordinal = []
        self.categorical_features_ordinal_number=0

        #Other features
        self.features_text = []
        self.features_text_number=len(self.features_text)
        self.features_removed = []
        self.features_removed_number=len(self.features_removed)


    def __update_XY(self):
        self.X=self.data_original[self.features]
        self.Y=self.data_original[self.targets]
        self.features_number=len(self.features)
        self.targets_number=len(self.targets)

        #Numerical features
        self.numerical_features_number = len(self.numerical_features)
        self.numerical_features_circular_number=len(self.numerical_features_circular)
        #Categorical features
        self.categorical_features_number=len(self.categorical_features)
        self.categorical_features_nominal_number = len(self.categorical_features_nominal)
        self.categorical_features_ordinal_number = len(self.categorical_features_ordinal)
        #Other features
        self.features_text_number=len(self.features_text)
        self.features_removed_number_number=len(self.features_removed)

    def __remove_label_from_lists(self,label):
        try:
            self.features.remove(label)
            self.numerical_features.remove(label)
            self.numerical_features_circular.remove(label)
            self.categorical_features.remove(label)
            self.categorical_features_nominal.remove(label)
            self.categorical_features_ordinal.remove(label)
            self.features_text.remove(label)
        except:
            pass

    def show_info(self):
        print('--------------------------------------')
        print('Dataset summary')
        print(f'Original number of columns in dataset: {self.columns_number}')
        print(f'Features (columns) discarded number: {self.features_removed_number_number}')
        print(f'Targets number: {self.targets_number}')
        print(f'Features number: {self.features_number}')
        print(f'\tNumerical features number: {self.numerical_features_number}')
        print(f'\t\tNumerical features number (circular): {self.numerical_features_circular_number}')
        print(f'\tCategorical features number: {self.categorical_features_number}')
        print(f'\t\t Categorical features number (nominal): {self.categorical_features_nominal_number}')
        print(f'\t\t Categorical features number (ordinal): {self.categorical_features_ordinal_number}')
        print(f'\tText features number: {self.features_text_number}')


    def define_targets(self, targets_list):
        for target in targets_list:
            self.__remove_label_from_lists(target)
            self.targets.append(target)

        self.__update_XY()

    def remove_features(self,features_list):
        self.features_removed=features_list
        for feature in features_list:
            self.__remove_label_from_lists(feature)
        self.__update_XY()

    def show_moments(self):
        #Function shows std, skewness and curtosis
        pass
    def show_cardinality(self):
        pass
    def show_missing_values(self):
        pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = DatasetKnowledge(csv_file_path='train.csv')

    print(dataset.features)
    dataset.define_targets(['SalePrice'])
    print(dataset.features)

    dataset.remove_features(['Id','MSSubClass'])
    print(dataset.targets)
    print(dataset.X.head(10))
    print(dataset.Y)
    dataset.show_info()
    print(f'Numerical values: {dataset.numerical_features}')
    print(f'Categorical values: {dataset.categorical_features}')