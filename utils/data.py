from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data():
    digits = datasets.load_digits()
    features = digits.images.reshape((digits.images.shape[0], -1))
    labels = digits.target
    data = train_test_split(features, labels, test_size=0.3, random_state=42)
    data = {
        'train': {
            'features': data[0],
            'labels': data[2],
        },
        'test': {
            'features': data[1],
            'labels': data[3]
        }
    }
    return data
