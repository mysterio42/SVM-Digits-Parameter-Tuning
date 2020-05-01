import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

FIGURES_DIR = 'figures/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def data_plot(digits):
    img_labels = list(zip(digits.images, digits.target))

    for i, (image, label) in enumerate(img_labels[::6]):
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f'Label: {label}')
        plt.show()


def plot_cm(cm, name):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.title(name)
    plt.savefig(FIGURES_DIR + f'Figure_{name}' + '.png')
    plt.show()


def plot_pca(data):
    pca = PCA(n_components=2)
    projections = pca.fit_transform(data['train']['features'])
    plt.scatter(projections[:, 0], projections[:, 1],
                c=data['train']['labels'], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Paired_r', 10))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig(FIGURES_DIR + 'Figure_digits' + '.png')
    plt.show()
