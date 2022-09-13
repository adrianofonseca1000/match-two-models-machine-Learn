# Importing the libraries
# Bibliotecas para criação e manipulação de DATAFRAMES e Algebra
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
# Internal libs
from Data.config import Base


class FeaturesModels(Base):
    def __init__(self, database):
        super().__init__(database)

    print("Loading Features...")

    def get_feature_train(self):
        self.train = pd.read_csv('treino.csv', encoding="latin-1", sep=";")
        return self.train

    def gen_features_train(self):
        self.test_size = 0.33
        self.random_state = 0

        # Loading dataset train
        print("Loading data Train...")
        """
        Retorna pandas dataframe
        """
        feature = self.get_feature_train()
        dataframe_train = pd.DataFrame(feature)

        # ___categorical feature
        print("Converting categorical feature")
        var_cat = dataframe_train.select_dtypes('object')
        for col in var_cat:
            dataframe_train[col] = LabelEncoder().fit_transform(dataframe_train[col].astype('str'))

        print("Convert to numpy array")
        features = dataframe_train.columns.difference(
            ['features', 'target'])
        self.Y = dataframe_train['venda'].values
        self.X = dataframe_train[features].values

        print("Train Test Split")
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)

    def _get_feature_test(self):
        # query test
        query = """
        query?
                """
        res = pd.read_sql(query, self.engine)
        return res

    def gen_features_test(self):
        # Loading dataset test
        print("Loading data Test...")
        """
        Retorna pandas dataframe
        """
        feature = self._get_feature_test()
        dataframe_test = pd.DataFrame(feature)

        # Dataset Phone
        self.telefone = dataframe_test.iloc[:, [0]]

        print("Convert to numpy array")
        self.target = dataframe_test.iloc[:, [11]]
        # print(self.target)
        self.feature_statsmodels = dataframe_test.iloc[:, [1, 2, 4, 7, 8, 9]]

        # ___categorical feature
        print("Converting categorical feature")
        var_cat = dataframe_test.select_dtypes('object')
        for col in var_cat:
            dataframe_test[col] = LabelEncoder().fit_transform(dataframe_test[col].astype('str'))

        self.feature_randomforest = dataframe_test.columns.difference(['feature', 'target'])
        self.feature_randomforest = dataframe_test[self.feature_randomforest].values
        # print(self.feature_randomforest)
        return dataframe_test
