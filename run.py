# Internal libs
from Data.data import FeaturesModels
from Models.models import Models

if __name__ == '__main__':
    data = FeaturesModels('base')
    data.gen_features_train()
    data.gen_features_test()

    print("Running Models")
    model = Models()
    model.model_statesmodel(data.get_feature_train())
    model.model_random_forest(data.X_train, data.Y_train, data.X_test, data.Y_test)

    print("Model Predict")
    model.loading_statesmodel(data.feature_statsmodels)
    model.loading_randomforest(data.feature_randomforest)
    model.soma(data.feature_statsmodels, data.feature_randomforest, data.telefone)
