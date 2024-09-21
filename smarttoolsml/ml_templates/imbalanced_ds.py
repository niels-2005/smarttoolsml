from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def random_over_sampler(X_train, y_train):
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)
    return X_train, y_train


def random_under_sampler(X_train, y_train):
    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(X_train, y_train)
    return X_train, y_train


def smote(X_train, y_train):
    smte = SMOTE()
    X_train, y_train = smte.fit_resample(X_train, y_train)
    return X_train, y_train