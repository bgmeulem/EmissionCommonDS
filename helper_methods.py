import warnings
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from imblearn.pipeline import Pipeline  # not sklearn.pipeline, as that one can't work with oversampling
from sklearn.metrics import ConfusionMatrixDisplay, plot_roc_curve
from category_encoders.woe import WOEEncoder
import seaborn as sb


class NamedWOEEncoder(WOEEncoder):
    """
    Wrapper class around WOEEncoder to have consistent naming convention for get_feature_names_out
    """
    def __init__(self):
        super().__init__()

    def get_feature_names_out(self, feature_names):
        return super().get_feature_names()


class RectifySkewedColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('RectifySkewedColumns initialised')
        self.transf = np.log
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns

    def transform(self, X, y=None, transf=np.log):
        X_ = X.copy()
        for col in X_.columns:
            if type(transf) == float:
                X_[col] = [e ** transf for e in X_[col].values]
            else:
                X_[col] = [transf(e) if e > 0 else np.nan for e in X_[col].values]
        return X_

    def fit_transform(self, X, y=None, transf=np.log):
        self.fit(X)
        return self.transform(X, y=None, transf=transf)

    def get_feature_names_out(self, *args):
        return self.columns


class HighCardTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, high_card_cols):
        self.high_card_cols = high_card_cols
        print("Initialised HighCardTransformer")

    def fit(self, x, y=None):
        x.drop(self.high_card_cols, axis=1, inplace=True)

    def transform(self, x, y=None):
        x.drop(self.high_card_cols, axis=1, inplace=True)


class OneHotEncoderContinuous(BaseEstimator, TransformerMixin):
    """
    Encodes to 1 if the continuous value is larger than (or equal to) 0, encodes to 0 if otherwise
    """

    def __init__(self):
        print("Initialised HighCardTransformer")
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns

    def transform(self, X, y=None):
        X_ = X.copy()
        for col in X_.columns:
            X_[col] = [1 if float(e) >= 0 else 0 for e in X_[col].values]
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self, *args):
        return self.columns


def plot_confusion_matrix(clf, model_name, X_test, y_test, score):
    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
        normalize='true',
    )
    disp.ax_.set_title("{}: {:.4f}\n".format(model_name, score))
    plt.subplots_adjust(.05, .15, .95, .75)
    disp.figure_.savefig("Plots/Confusion_Matrix_{}".format(model_name))
    plt.show(block=False)
    plt.pause(.2)
    plt.close()


def plot_roc(clf, model_name, X_test, y_test):
    disp = plot_roc_curve(clf, X_test, y_test)
    disp.figure_.subplots_adjust(0.15, 0.15, 1, .9)
    disp.ax_.plot([0, 1], [0, 1], zorder=-1, linestyle='--')
    disp.figure_.suptitle("ROC curve for {}".format(model_name))
    disp.figure_.savefig("Plots/ROC_{}.png".format(model_name), dpi=300)
    plt.show(block=False)
    plt.pause(.2)
    plt.close()


def make_prediction(clf, processing_pipeline, model_name):
    pred_df = pd.read_excel('Data/DSC_2021_Test.xlsx').drop(['Label_Default'], axis=1)
    pred_df = processing_pipeline.transform(pred_df)
    pred = clf.predict_proba(pred_df)
    pred = [p[1] for p in pred]
    pd.DataFrame({'scores': pred}, index=range(len(pred))).to_csv('Model_Info/predictions_{}.csv'.format(model_name),
                                                                  index=False)


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    DEPRECATED, since sklearn >= 1.1 has added the get_feature_names_out() method to all transformers
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    # TODO: for small samples, sometimes off-by-one error. Probably due to not having all
    #  values of some categorical feature?
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


def plot_importance(df, model_name):
    ax = df.iloc[:10].plot.barh(x='feature', y='importance')  # plot top 10 most important features
    ax.legend().remove()
    ax.set_yticklabels(
        textwrap.fill(str(e).split('__')[-1][:-2], 28) for e in ax.get_yticklabels())  # remove processing prefixes
    plt.subplots_adjust(.65, .15, .9, .9)
    plt.xlabel('Importance')
    plt.suptitle('Feature importance for ' + model_name)
    ax.get_figure().savefig("Plots/Importance_{}".format(model_name))
    plt.show(block=False)
    plt.pause(.2)
    plt.close()


def split_train_test(f="Data/DSC_2021_Training.xlsx", train_size=.8, sample=None):
    df = pd.read_excel(f)
    if sample is not None:
        df = df.sample(sample)
    X, y = df.drop("Label_Default", axis=1), df["Label_Default"]
    y = [1 if e == 'Y' else 0 for e in df["Label_Default"].values]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=10)
    X_train.to_csv("Data/Train.csv")
    X_test.to_csv("Data/Test.csv")
    return X_train, X_test, y_train, y_test


def get_skewed_cols(df, max_skew):
    skewness = df.skew(axis=0, skipna=True, numeric_only=True).sort_values(ascending=False) \
        .to_frame(name='skew').reset_index()
    SKEWED_COLS = skewness[abs(skewness['skew']) >= max_skew]['index'].values
    return SKEWED_COLS


def fix_skewness(df, max_skew, transf=.5):
    skewness = df.skew(axis=0, skipna=True, numeric_only=True).sort_values(ascending=False) \
        .to_frame(name='skew').reset_index()
    cols = skewness[abs(skewness['skew']) >= max_skew]['index'].values
    for col in cols:
        if type(transf) == float:
            df[col] = [e ** transf for e in df[col].values]
        else:
            df[col] = [transf(e) if e > 0 else np.nan for e in df[col].values]
    return df


def show_importance(clf, model_name, colnames):
    # colnames = get_feature_names(preprocessing_transformers)
    if model_name == 'rf':
        importances = clf.feature_importances_
    elif model_name == 'lr' or model_name == 'lr_dummy':
        importances = clf.coef_[0]
    else:
        importances = None
        fn = None
    if importances is not None:
        imp = pd.DataFrame({'feature': colnames, 'importance': importances}).reset_index() \
            .sort_values('importance', key=abs, ascending=False)
        imp.to_csv('Model_Info/feature_importance_{}.csv'.format(model_name))
        plot_importance(imp, model_name)
        print("#########################################")


def plot_distribution_of_features(X_):
    """
    A function that takes a dataset in pd.DataFrame format and plots out
    a distribution of each column. Also calculates the skewness.
    Only tested for continuous variables. Will probably need to be adapted
    for categorical variables (just change histplot to barplot)

    Args:
    X: pd.DataFrame: the dataframe

    Returns: 0 if success
    """
    NOMINAL = ['Type', 'Managing_Sales_Office_Nbr', 'Postal_Code_L',
               'Product_Desc', 'CREDIT_TYPE_CD', 'FINANCIAL_PRODUCT_TYPE_CD',
               'INDUSTRY_CD_3', 'INDUSTRY_CD_4',
               'ACCOUNT_PURPOSE_CD', 'A2_MARITAL_STATUS_CD', 'A2_EMPLOYMENT_STATUS_CD',
               'A2_RESIDENT_STATUS_CD']  # features that will get normal nominal treatment, hardcoded for now: TODO
    for feature_name in X_:  # loop over the names of the columns
        values = X_[feature_name]  # get the values of the feature
        # skewness_ = stats.skew(values, nan_policy="omit")  # while we're at it, calculate the skewness as well
        plt.clf()
        if feature_name in NOMINAL:
            cnt = values.value_counts()
            ax_ = sb.barplot(x=cnt.index.values, y=cnt.values)
        else:
            ax_ = sb.histplot(values.astype(float),  # the target variables
                              kde=True  # show a smooth trend line
                              )
        ax_.set_xlabel(feature_name)
        ax_.get_figure().suptitle("{}".format(feature_name))
        plt.show(block=False)
        plt.pause(.05)
    plt.close('all')
    return 0