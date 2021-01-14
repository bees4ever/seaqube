"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

from functools import reduce
import spacy
from pyinflect import getAllInflections

from seaqube.augmentation.base import SingleprocessingAugmentation
from seaqube.nlp.tools import tokenize_corpus
from seaqube.package_config import log


class Tenses:
    def __init__(self):
        props = ['SIMPLE_PRESENT', 'PRESENT_PROGRESSIVE', 'PRESENT_PERFECT', 'PAST_PROGRESSIVE', 'SIMPLE_PAST',
                 'PAST_PERFECT',
                 'PRESENT_PERFECT_PROGRESSIVE', 'PAST_PERFECT_PROGRESSIVE', 'FUTURE_I_WILL', 'FUTURE_I_GOING',
                 'FUTURE_I_PROGRESSIVE', 'FUTURE_II_PROGRESSIVE', 'FUTURE_II',
                 'CONDITIONAL_I',  # can not be detected
                 'CONDITIONAL_I_PROGRESSIVE',  # is not working as for progressive
                 'CONDITIONAL_II',
                 'CONDITIONAL_II_PROGRESSIVE'  # is not working as for progressive
                 ]
        for prop in props:
            self.__dict__[prop] = prop


tenses = Tenses()


class Active2PassiveAugmentation(SingleprocessingAugmentation):
    """
    Rules for active to passive including backshifting and more:

    - https://www.ego4u.com/en/cram-up/grammar/tenses
    - https://github.com/explosion/spaCy/issues/2767
    - https://www.edudose.com/english/active-and-passive-voice-rules/



    """

    def input_type(self):
        """
        Which return type is supported
        Returns: doc or text
        """
        return "text"

    def __init__(self, remove_duplicates: bool = False, original_too: bool = False, multiprocess: bool = True, seed: int = None):
        """
        Set up the spacy backend to use it for NER
        Args:
            remove_duplicates: remove after augmentation for duplicates
            original_too: if the original input should be added to the result
              multiprocess: if augmentation class implements the multiprocessing call, then it can be turn off again with
                    this flag, most for testing purpose
            seed: fix the randomness with a seed for testing purpose
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError(
                "Spacy Pretrained Model is not available, please run: `from seaqube import download;download('spacy-en-pretrained')`")

        self.remove_duplicates = remove_duplicates
        self.seed = seed
        self.tenses = Tenses()
        self.multiprocess = multiprocess
        self.original_too = original_too
        super(Active2PassiveAugmentation, self).__init__()

    def get_config(self):
        """
        Gives a dict with all relevant variables the object can recreated with (init parameters)
        Returns: dict of object config

        """
        return dict(remove_duplicates=self.remove_duplicates, original_too=self.original_too, seed=self.seed, spacy=str(self.nlp), class_name=str(self))

    def shortname(self):
        return "active2passive"

    def augmentation_implementation(self, text):
        parts = []
        for sent in self.nlp(text).sents:
            try:
                parts.append(self.sentence2passive(str(sent)))
            except ValueError as ex:
                log.info(
                    f"A value error raised with following content: {str(ex)} and traceback: {str(ex.__traceback__)}")
        result = []
        if len(parts) > 0:
            result = [reduce(lambda a, b: a + b, parts)]

        if self.original_too:
            result.append(tokenize_corpus([text], verbose=False)[0])

        return result

    def get_sentene_tense(self, tokens):
        """
        Detect the tense based on one sentence (no nested sentences allowed)
        """
        tags = [d.tag_ for d in tokens]

        if ('VBP' in tags and 'VBN' in tags and 'VBG' in tags) or (
                'VBZ' in tags and 'VBN' in tags and 'VBG' in tags):
            return self.tenses.PRESENT_PERFECT_PROGRESSIVE

        if ('VBD' in tags and 'VBN' in tags and 'VBG' in tags):
            return self.tenses.PAST_PERFECT_PROGRESSIVE

        if ('VBZ' in tags and 'VBG' in tags and 'TO' in tags and 'VB' in tags) or (
                'VBP' in tags and 'VBG' in tags and 'TO' in tags and 'VB' in tags):
            return self.tenses.FUTURE_I_GOING

        if ('VBZ' in tags and 'VBG' in tags) or ('VBP' in tags and 'VBG' in tags):
            return self.tenses.PRESENT_PROGRESSIVE

        if 'VBD' in tags and 'VBG' in tags:
            return self.tenses.PAST_PROGRESSIVE

        if 'VBD' in tags and 'VBN' in tags:
            return self.tenses.PAST_PERFECT

        if 'VBZ' in tags and 'VBN' in tags or ('VBP' in tags and 'VBN' in tags):
            return self.tenses.PRESENT_PERFECT

        if 'VBZ' in tags or ('VBZ' in tags and 'VB' in tags) or 'VBP' in tags or ('VBP' in tags and 'VB' in tags):
            return self.tenses.SIMPLE_PRESENT

        if 'VBD' in tags or ('VBD' in tags and 'VB' in tags):
            return self.tenses.SIMPLE_PAST

        if 'MD' in tags and 'VB' in tags and 'VBN' in tags and 'VBG' in tags:
            return self.tenses.FUTURE_II_PROGRESSIVE

        if 'MD' in tags and 'VB' in tags and 'VBN' in tags:
            return self.tenses.FUTURE_II

        if 'MD' in tags and 'VB' in tags and 'VBG' in tags:
            return self.tenses.FUTURE_I_PROGRESSIVE

        if 'MD' in tags and 'VB' in tags and 'VBN' in tags:
            return self.tenses.FUTURE_II

        if 'MD' in tags and 'VB' in tags:
            return self.tenses.FUTURE_I_WILL

    def verb_to_participe(self, token):
        word = self.nlp.vocab[token.lemma].text
        log.debug(f"[verb_to_participe]: inflections {word}")
        try:
            vbn = getAllInflections(word)['VBN']
            return vbn[0]
        except KeyError:
            return token

    @staticmethod
    def be_present(person, singular):
        person = str(person).lower()

        if not singular:
            return 'are'

        if person in ['you', 'we', 'they']:
            return 'are'
        if person in ['i']:
            return 'am'

        return 'is'

    @staticmethod
    def have_present(person, singular):
        person = str(person).lower()

        if not singular:
            return 'have'
        else:
            if person in ['i', 'you', 'we', 'they']:
                return 'have'
            return 'has'

    @staticmethod
    def be_past(person, singular):
        person = str(person).lower()

        if not singular:
            return 'were'
        else:
            if person in ['you', 'we', 'they']:
                return 'were'
            if person in ['i']:
                return 'was'

            return 'was'

    def backshift(self, tense, verb, singular, person=None):
        participe = self.verb_to_participe(verb)

        if tense == tenses.SIMPLE_PRESENT:
            return [self.be_present(person, singular), participe]

        if tense == tenses.PRESENT_PROGRESSIVE:
            return [self.be_present(person, singular), 'being', participe]

        if tense == tenses.PRESENT_PERFECT:
            return [self.have_present(person, singular), 'been', participe]

        if tense == tenses.PRESENT_PERFECT_PROGRESSIVE:
            return [self.have_present(person, singular), 'been', 'beeing', participe]

        if tense == tenses.SIMPLE_PAST:
            return [self.be_past(person, singular), participe]

        if tense == tenses.PAST_PROGRESSIVE:
            return [self.be_past(person, singular), 'being', participe]

        if tense == tenses.PAST_PERFECT:
            return ['had', 'been', participe]

        if tense == tenses.PAST_PERFECT_PROGRESSIVE:
            return ['had', 'been', 'beeing', participe]

        if tense == tenses.FUTURE_I_WILL:  # could be also Conditional I Simple and more
            return ['be', participe]

        if tense == tenses.FUTURE_I_GOING:
            return [self.be_present(person, singular), 'going', 'to', 'be', participe]

        if tense == tenses.FUTURE_II:
            return ['have', 'been', participe]  # could be also Conditional II Simple and more

        log.debug(f"Backshift failed, tense={tense}")
        raise ValueError("Backshift failed due to complexity")

    @staticmethod
    def nouninv(noun):
        noundict = {'i': 'me', 'we': 'us', 'you': 'you', 'he': 'him', 'she': 'her', 'they': 'them', 'them': 'they',
                    'her': 'she', 'him': 'he', 'us': 'we', 'me': 'i'}
        n = noun.lower()
        if n in noundict:
            return noundict[n]
        return noun

    @staticmethod
    def neg_expand(short_neg):
        neg = str(short_neg).strip()
        tr = {
            "n't": 'not'
        }

        if neg in tr.keys():
            return tr[neg]
        else:
            return short_neg

    def sentence2passive(self, text):
        """
        https://stackoverflow.com/q/55086029/5885054
        In order to modify the verb forms, you need a morphological analyzer (makes in base form -> make) and a morphological generator (base form make as past participle -> made, as present participle -> making). Spacy can do the analysis step for English, but not the generation step, so you'll need to look for additional tools.

        https://www.learncbse.in/cbse-class-10-english-grammar-active-passive-voice/
        https://resources.german.lsa.umich.edu/grammatik/active-vs-passive/
        https://owl.purdue.edu/owl/general_writing/academic_writing/active_and_passive_voice/changing_passive_to_active_voice.html


        """

        doc = self.nlp(text)

        doc_tense = self.get_sentene_tense(doc)
        log.debug(f"[sentence2passive] tense={doc_tense}")

        if doc_tense in [tenses.FUTURE_I_PROGRESSIVE, tenses.FUTURE_II_PROGRESSIVE, tenses.PAST_PERFECT_PROGRESSIVE,
                         tenses.PRESENT_PERFECT_PROGRESSIVE]:
            raise ValueError(f"The given tense ({doc_tense}) can not changed into passive voice")

        tags = [d.tag_ for d in doc]
        deps = [d.dep_ for d in doc]
        poses = [d.pos_ for d in doc]

        log.debug(f"[sentence2passive] doc={str(doc)}")
        log.debug(f"[sentence2passive] tags={str(tags)}")
        log.debug(f"[sentence2passive] deps={str(deps)}")
        log.debug(f"[sentence2passive] poses={str(poses)}")

        used_indecies = []

        passived_sentence = []

        singular = True
        # if 'NOUN' in poses:
        if 'dobj' in deps:
            # the NOUN is the new "start" point
            noun_pos = deps.index('dobj')

            ####### GO as long back as a articel is there (under part need to checked)
            backwards_index = noun_pos - 1
            backwards_index_selected = []

            while poses[backwards_index] in ['ADV', 'CCONJ', 'ADJ']:
                backwards_index_selected.append(backwards_index)
                backwards_index -= 1


            if poses[backwards_index] == 'DET' or poses[noun_pos] == 'PROPN':
                while poses[backwards_index] == 'DET':
                    backwards_index_selected.append(backwards_index)
                    backwards_index -= 1

                backwards_index += 1  # Go one step back, while has ended and there is nothing correct anymore

                # we have a big list of stuff to added before object
                det = doc[backwards_index]
                if deps[backwards_index] == 'poss':
                    det = 'the'

                backwards_index_selected.reverse()

                before_noun = [doc[j] for j in backwards_index_selected]
                if len(before_noun) > 0:
                    before_noun[0] = det

                log.debug(f"[sentence2passive] found articles with appendix={str(before_noun)}")

                passived_sentence += before_noun
                used_indecies += backwards_index_selected
                used_indecies.append(backwards_index)

            passived_sentence.append(self.nouninv(str(doc[noun_pos])))
            used_indecies.append(noun_pos)



        elif 'pobj' in deps:
            noun_pos = deps.index('pobj')

            backwards_index = noun_pos-1
            backwards_index_selected = []
            ## this is not a person, but a positional object (accusativ, dativ) I don't now, however, maybe with article
            while poses[backwards_index] in ['ADV', 'CCONJ', 'ADJ']:
                backwards_index_selected.append(backwards_index)
                backwards_index -= 1

            if poses[backwards_index] == 'DET':
                while poses[backwards_index] == 'DET':
                    backwards_index_selected.append(backwards_index)
                    backwards_index -= 1

            backwards_index += 1  # Go one step back, while has ended and there is nothing correct anymore

            # we have a big list of stuff to added before object
            det = doc[backwards_index]
            if deps[backwards_index] == 'poss':
                det = 'the'

            backwards_index_selected.reverse()

            before_noun = [doc[j] for j in backwards_index_selected]
            if len(before_noun) > 0:
                before_noun[0] = det

            log.debug(f"[sentence2passive] found articles with appendix={str(before_noun)}")

            passived_sentence += before_noun
            used_indecies += backwards_index_selected
            used_indecies.append(backwards_index)

            # there could be some article and even adjectives in between (I think)
            passived_sentence.append(doc[noun_pos])
            used_indecies.append(noun_pos)
            # also remove a possible ADP before
            if poses[backwards_index - 1] == 'ADP':
                used_indecies.append(backwards_index - 1)
            # also remove a possible ADP before
            if poses[backwards_index - 1] == 'DET' and poses[backwards_index - 2] == 'ADP':
                used_indecies.append(backwards_index - 2)


        else:
            raise ValueError(
                "This sentence can not transformed to passive voice or this sentence structure is too complex for this implementation")

        if tags[noun_pos] in ['NNS']:
            singular = False

        ## Select the VERB

        if 'AUX' in poses and 'VERB' in poses:
            aux_pos = poses.index('AUX')
            # passived_sentence.append(doc[aux_pos])
            used_indecies.append(aux_pos)
            if len(poses) > aux_pos + 1 and poses[aux_pos + 1] == "AUX":  ## double aux is already done in backshift
                used_indecies.append(aux_pos + 1)

        if 'VERB' in poses:
            if doc_tense in [tenses.FUTURE_I_WILL, tenses.FUTURE_II]:
                verb_pos = poses.index('VERB')
                while deps[verb_pos] == 'aux':
                    used_indecies.append(verb_pos)
                    verb_pos = poses.index('VERB', verb_pos + 1)

            elif doc_tense in [tenses.FUTURE_I_GOING]:
                verb_pos = poses.index('VERB')
                if poses[verb_pos + 1] == 'PART' and poses[verb_pos + 2] == 'VERB':
                    used_indecies.append(verb_pos)
                    used_indecies.append(verb_pos + 1)
                    verb_pos = verb_pos + 2


                else:
                    raise ValueError("Provided Sentence is not in a correct `Future I - going to` form")


            else:
                verb_pos = poses.index('VERB')

            verbs = self.backshift(doc_tense, doc[verb_pos], singular, person=passived_sentence[0])

            used_indecies.append(verb_pos)

            if doc_tense in [tenses.FUTURE_I_WILL, tenses.FUTURE_II]:  # this is the same tense as for condition I
                verbs.insert(0, doc[tags.index('MD')])

            if deps[verb_pos - 1] in ['advmod', 'neg']:
                verbs.insert(1, self.neg_expand(doc[verb_pos - 1]))
                used_indecies.append(verb_pos - 1)

            if deps[verb_pos - 1] == 'aux' and deps[verb_pos - 2] in ['advmod', 'neg']:
                verbs.insert(1, self.neg_expand(doc[verb_pos - 2]))
                used_indecies.append(verb_pos - 2)
                used_indecies.append(verb_pos - 1)

            passived_sentence += verbs


        elif 'AUX' in poses:
            verb_pos = poses.index('AUX')

            if len(poses) > verb_pos + 1 and poses[verb_pos + 1] == "AUX":  ## double aux is already done in backshift
                used_indecies.append(verb_pos + 1)

            passived_sentence += self.backshift(doc_tense, doc[verb_pos], singular, person=passived_sentence[0])

            used_indecies.append(verb_pos)
        else:
            raise ValueError("No verb nor aux in sentence, I can not tranform it")

        ## Select the SUBJECT

        subject_pos = deps.index('nsubj')
        used_indecies.append(subject_pos)
        by_objects = ['by']
        if poses[subject_pos - 1] == 'DET':
            by_objects.append(doc[subject_pos - 1].text.lower())
            used_indecies.append(subject_pos - 1)

        by_objects.append(self.nouninv(doc[subject_pos].text))

        passived_sentence += by_objects

        # check the remaining words
        remaining = list(set(range(len(doc))).difference(set(used_indecies)))
        remaining.sort()

        for j in remaining:
            passived_sentence.append(doc[j])

        # fix captials
        passived_sentence = list(map(lambda d: str(d), passived_sentence))
        passived_sentence[0] = passived_sentence[0].capitalize()

        return passived_sentence

    # TODO split if conditions and put them first # if there is nothing else"
