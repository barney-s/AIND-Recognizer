import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError


    def base_model(self, num_states, X=None, lens=None, testX=None, testlens=None):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        if X is None:
            X = self.X
        if lens is None:
            lens = self.lengths
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lens)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))

            if testX is not None:
                return hmm_model, hmm_model.score(testX, testlens)
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        logN = np.log(len(self.X))
        least = float("inf")
        for count in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(num_states=count)
            try:
                logL = model.score(self.X, self.lengths)
                p = model.n_features
                bic = -2 * logL + p * logN
                if bic < least:
                    least = bic
                    best = model
            except:
                pass
        return best


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        highest = float("-inf")
        for count in range(self.min_n_components, self.max_n_components + 1):
            dic = None
            try:
                model = self.base_model(num_states=count)
                score = model.score(self.X, self.lengths)
                score_sum = 0.0
                for w, xl in self.hwords.items():
                    x, l = xl[0], xl[1]
                    if w != self.this_word:
                        score_sum += model.score(x, l)
                mean = score_sum/(len(self.hwords)-1)
                dic = score - mean
            except:
                continue
            if dic > highest:
                best = model
                highest = dic
        return best


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        highest = float("-inf")
        best = self.n_constant
        for count in range(self.min_n_components, self.max_n_components+1):
            k = 3
            if(len(self.sequences) < k):
                break
            split = KFold(n_splits=k, shuffle=False,
                          random_state=self.random_state)
            scores = []
            for train, test in split.split(self.sequences):
                trX, trLens = combine_sequences(train, self.sequences)
                tsX, tsLens = combine_sequences(test, self.sequences)
                try:
                    _, score = self.base_model(count, X=trX, lens=trLens,
                                               testX=tsX, testlens=tsLens)
                    scores.append(score)
                except:
                    pass
            avg = np.average(scores)
            if avg > highest:
                highest = avg
                best = count
        return self.base_model(best)
