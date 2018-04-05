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

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def get_bic_score(self, num_components):
        # Formula: BIC = −2 log L + p log N
        # L =likelihood of the fitted model
        # p =number of parameters
        # N =num of data points
        
        bic_model = self.base_model(num_components)
        N = len(self.sequences)
        p = (num_components**2 + 2*num_components*bic_model.n_features - 1)
        logL = bic_model.score(self.X, self.lengths)
        bic = -2*logL + p*math.log(N)
        return bic, bic_model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        ##Init
        best_bic = float("inf") #lower the better
        best_bic_model = None
        
        ##loop through all components
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                temp_bic, temp_bic_model = self.get_bic_score(num_components)
                if temp_bic < best_bic:
                    best_bic = temp_bic
                    best_bic_model = temp_bic_model
            except:
                continue
        return best_bic_model 
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def get_dic_score(self, num_components):
        """
        Calculates the avg log likelihood of cv folds
        
        """
        # Formula ->
        # DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
        
        
        dic_model = self.base_model(num_components)
        logL = dic_model.score(self.X, self.lengths)
        
        likelihood_all_but_current_word = []
        for current_word, (X, lengths) in self.hwords.items():
            if current_word != self.this_word:
                likelihood_all_but_current_word.append(dic_model.score(X, lengths))
        dic_score = logL -  np.mean(likelihood_all_but_current_word)
        return dic_score, dic_model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        ## Logic similar to BIC selector
        best_dic = float("-inf")
        best_dic_model = None
        
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                temp_dic, temp_dic_model = self.get_dic_score(num_components)
                if temp_dic > best_dic:
                    best_dic = temp_dic
                    best_dic_model = temp_dic_model
            except:
                continue
        return best_dic_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def get_cv_score(self, num_components):
        
        
        scores = []
        split_method= KFold()
        
        for train_index,test_index in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_index, self.sequences)
            test_X, test_length = combine_sequences(test_index, self.sequences)
            model = self.base_model(num_states=num_components).fit(self.X, self.lengths)
            
            scores.append(model.score(test_X, test_length))
        
        cv=np.mean(scores)

        return cv, model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        

        
        try:
            best_cv = float("-inf") ## higher the better
            model = None
            
            for num_components in range(self.min_n_components, self.max_n_components + 1):
                temp_cv, temp_model= self.get_cv_score(num_components)
                if temp_cv > best_cv:
                    best_cv = temp_cv
                    model = temp_model

            return model
        except:
            return self.base_model(self.n_constant)
