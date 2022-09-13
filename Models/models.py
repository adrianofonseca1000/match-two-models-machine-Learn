import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle


class Models(object):
    def model_statesmodel(self, train):
        self.train = train
        model_statsmodels = smf.glm(
            'target ~ features',
            data=self.train,
            family=sm.families.Binomial()).fit()
        print("Saving Train LR")
        filename = 'LogisticRegression.sav'
        pickle.dump(model_statsmodels, open(filename, 'wb'))
        return model_statsmodels

    def model_random_forest(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        model_randomforest = RandomForestClassifier(random_state=42, bootstrap=True, max_depth=80, max_features=2,
                                                    min_samples_leaf=5, min_samples_split=8, n_estimators=100).fit(
            self.X_train,
            self.Y_train)
        self.Y_pred = model_randomforest.predict(self.X_test)

        print("Saving Train RF")
        filename = 'RandomForest.sav'
        pickle.dump(model_randomforest, open(filename, 'wb'))

        if self.model_random_forest is None:
            print("Modelo não carregado.")
            return
        print("Métricas Training Random Forest:")
        print("Accuracy:", accuracy_score(self.Y_test, self.Y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(self.Y_test, self.Y_pred))
        print("Classification report:")
        print(classification_report(self.Y_test, self.Y_pred))
        return model_randomforest

    def threshold_statsmodels(self, i):
        if i['prop_log'] < 0.5:
            return 0
        else:
            return 1

    def loading_statesmodel(self, feature_statsmodels):
        # self.venda=venda
        self.feature_statsmodels = feature_statsmodels
        logistic = joblib.load('LogisticRegression.sav')
        predict = logistic.predict(self.feature_statsmodels)
        self.feature_statsmodels['prop_log'] = predict
        # print("Saving Score States Model")
        self.feature_statsmodels['glm_bin_class'] = self.feature_statsmodels.apply(self.threshold_statsmodels, axis=1)
        self.score_statsmodels = pd.DataFrame(self.feature_statsmodels['glm_bin_class'])
        # print(self.score_statsmodels)
        # self.score_statsmodels = pd.concat([pd.DataFrame(telefone), pd.DataFrame(self.score_statsmodels)], axis=1).to_csv('prop_LogisticRegression.csv', sep=';', index=False)
        # print(classification_report(self.venda, self.feature_statsmodels['glm_bin_class']))
        # print(confusion_matrix(self.venda, self.feature_statsmodels['glm_bin_class']))
        return self.score_statsmodels

    def loading_randomforest(self, feature_randomforest):
        self.feature_randomforest = feature_randomforest
        randomforest = joblib.load('RandomForest.sav')
        predict = randomforest.predict(self.feature_randomforest)
        # print(classification_report(self.venda, predict))
        # print(confusion_matrix(self.venda, predict))
        # print(self.feature_randomforest.info())
        # print("Saving Score Random Forest")
        # print(predict)
        self.score_radomforest = pd.DataFrame(predict, columns=['randomforest'])
        # self.score_radomforest = pd.concat([pd.DataFrame(telefone), pd.DataFrame(self.score_radomforest)], axis=1).to_csv('prop_RandomForest.csv', sep=';', index=False)
        return self.score_radomforest

    def propension(self, i):
        if i['soma'] == 2:
            return 'AP'
        elif i['soma'] == 1:
            return 'MP'
        else:
            return 'BP'

    def soma(self, feature_statsmodels, feature_randomforest, telefone):
        self.feature_statsmodels = feature_statsmodels
        self.feature_randomforest = feature_randomforest
        self.telefone = telefone
        df = pd.concat(
            [self.loading_statesmodel(self.feature_statsmodels), self.loading_randomforest(self.feature_randomforest)],
            axis=1)
        df['soma'] = df['glm_bin_class'] + df['randomforest']
        df['Propensao'] = df.apply(self.propension, axis=1)
        score = pd.DataFrame(df['Propensao'])
        print(df['Propensao'].value_counts())
        score = pd.concat([pd.DataFrame(self.telefone), pd.DataFrame(score)], axis=1).to_csv('Propensão.csv',
                                                                                             sep=';', index=False)
        return score
