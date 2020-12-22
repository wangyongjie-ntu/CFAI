#Filename:	test_plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 13 Des 2020 02:50:08  WIB

from model.nn import NNModel
from cf.plainCF import PlainCF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.adult_dataset import load_adult_income

if __name__ == "__main__":

    income_df = load_adult_income("data/adult/adult.csv")
    d = Dataset(dataframe = income_df, continuous_features = ['age', 'education', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], outcome_name = 'income', scaler = MinMaxScaler())
    clf = NNModel(model_path = 'weights/adult.pth')
    cf = PlainCF(d, clf)
    test_instance = {'age': 57, 'workclass' : 'Self-Employed', 'education' : 2, 'educational-num': 10, 'marital-status':'Married',
            'occupation':'Service',  'relationship':'Husband', 'race':'White', 'gender':'Male',
            'capital-gain':0, 'capital-loss':0, 'hours-per-week':60, 'native-country':1}
    
    results = cf.generate_counterfactuals(test_instance, features_to_vary = ['education', 'educational-num', 'capital-gain', 'capital-loss',
        'hours-per-week'])
    results = results.detach().numpy()
    results = d.denormalize_data(results)
    results = d.onehot_decode(results)
    print(results) 

