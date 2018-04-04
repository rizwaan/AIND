import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
   
    
    all_seq = test_set.get_all_sequences()
    
    for index, seq in all_seq.items():
        X, lengths= test_set.get_item_Xlengths(index)
        guess_dict = {}

        best_guess_word = None
        best_score = float("-inf")

        for word, model in models.items():
            try:
                temp_score= model.score(X, lengths)
                #print("{} : {}".format(word,temp_score))
                guess_dict[word]= temp_score
                if temp_score > best_score:
                    best_score = temp_score
                    best_guess_word = word
                #print(models.items())

            except:
                guess_dict[word]= float('-inf')

        guesses.append(best_guess_word)
        probabilities.append(guess_dict)

    return probabilities, guesses
