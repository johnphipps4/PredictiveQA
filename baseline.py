import time
start_time = time.time()
from tqdm import tqdm
import spacy
from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from spacy.tokens import Token
import json

nlp = spacy.load('en_core_web_sm')

question_pronoun = {"what":[""], "how":[""], "why":[""], "which":[""], "when": ["DATE","TIME"], "who":["PERSON"],
"where":["LOC","GPE"], "whose":["PERSON"], "whom":["PERSON"]} #considering reasoning and description
                                                # is much harder
question_pronoun_known = ["when", "who", "where", "whose", "whom"]
def isMatch(q_head,ch_head):
    for x in (wn.synsets(q_head)):
        if wn.synsets(ch_head) != []:
            lc_hypers = x.lowest_common_hypernyms((wn.synsets(ch_head)[0]))
            if lc_hypers != []:
                if lc_hypers[0] == x:
                    return True
    return False

# helper function to determine whether or not getSimArray passes test
def passSimTest(arr):
    thresh = 0.75
    return (arr[0] > thresh and (arr[1] > thresh or arr[2] > thresh))

def isSim(q_head,ch_head):
    heads = []
    if (wn.synsets(q_head) != [] and wn.synsets(ch_head) != []):
        # get main question head similarity
        synset1 = wn.synsets(q_head)[0]
        synset2 = wn.synsets(ch_head)[0]
        val_main = synset1.wup_similarity(synset2)
        if val_main is None: val_main = 0
        heads.append(val_main) # add main score to output array

        # get largest hypo question head similarity (of top 5 hyponyms)
        top_hypo_val = 0
        hypo_count = 0
        for x in (synset1.hyponyms()):
            if hypo_count < 6:
                val_hypo = x.wup_similarity(synset2)
                if val_hypo is None: val_hypo = 0
                if val_hypo > top_hypo_val: top_hypo_val = val_hypo
            else: break
            hypo_count += 1
        heads.append(top_hypo_val) # add largest hyponym score to output array

        # get largest hyper question head similarity (of top 5 hypernyms)
        top_hyper_val = 0
        hyper_count = 0
        for x in (synset1.hypernyms()):
            if hyper_count < 6:
                val_hyper = x.wup_similarity(synset2)
                if val_hyper is None: val_hyper = 0
                if val_hyper > top_hyper_val: top_hyper_val = val_hyper
            else: break
            hyper_count += 1
        heads.append(top_hyper_val) # add largest hypernym score to output array
        return passSimTest(heads)
    else:
        return False


def get_NPs(doc):
    """ takes in a sentence string and returns a list of lists
        of noun_phrases [text_list, root_list, list_dep]
    """
    chunk_list_text = []
    chunk_list_root = []
    chunk_list_dep  = []
    chunk_list_fin  = []

    for noun in doc.noun_chunks:
        chunk_list_text.append(noun.text)
        chunk_list_root.append(noun.root.text)
        # chunk_list_dep.append(noun.root.dep_)
    chunk_list_fin.append(chunk_list_text)
    chunk_list_fin.append(chunk_list_root)
    # chunk_list_fin.append(chunk_list_dep)
    return chunk_list_fin

def get_Entities(doc):
    """ returns
    """
    ent_list_text  = []
    ent_list_label = []
    for entities in doc.ents:
        ent_list_text.append(entities.text)
        ent_list_label.append(entities.label_)
    ent_list_fin = []
    ent_list_fin.append(ent_list_text)
    ent_list_fin.append(ent_list_label)
    return ent_list_fin

def get_POS(doc):
    pos_list = []
    pos_list_s = []
    for token in doc:
        pos_list.append(token)
        pos_list_s.append(token.pos_)
    # t = list(zip(pos_list,pos_list_s))
    t=[pos_list,pos_list_s]
    return t

def get_NP_POS(doc):
    pos_list = []
    pos_list_s = []
    for token in doc:
        if (token.pos_  == "PROPN" or token.pos_ == "NOUN"):
            pos_list.append(token)
            pos_list_s.append(token.pos_)
    return (list(zip(pos_list,pos_list_s)))


def get_Wh(doc):
    """return the interrogative pronoun"""

    for elem in doc:
        if elem.text.lower() == "how":
            return "how"
        if (elem.pos_ is "NOUN" or elem.pos_ is "ADJ"):
            try:
                x = (elem.lemma_[:2]).lower()
                if(x == "wh"):
                    return elem.lemma_
            except:
                pass
    return "none"

def chunk_similarity(doc, qdoc):
    top_score = .65
    for qc in qdoc.noun_chunks:
        # chunk = qc.root.text
        for i in doc.noun_chunks:
            # chunk2 = i.root.text
            sim = (qc.similarity(i))
            if sim > top_score:
                return True
            # print("CHUNKS:")
            # print(qc.text)
            # print(i.text)
            # print(sim)
            # print("___________")
    return False

def map_Wh(q):
    "returns the expected POS list for the interrogative pronoun"
    # q_pronoun = get_Wh(q)
    if q in question_pronoun:
        return question_pronoun[q]
    return q

def get_wh_index(lst, q_val):
    """
    lst=chunked nouns
    q_val=pronoun
    returns index of the interrogative pronoun"""
    # print("------------GET_WH------------")
    # print(lst)
    # print(q_val)
    inner_index = 0
    if "ow" in q_val:
        q_val_upper = "How"
    else:
        q_val_upper = 'W'+q_val[1:]
        try:
            inner_index = [idx for idx, s in enumerate(lst) if (q_val in s or q_val_upper in s)][0]
        except:
            inner_index=0
    return inner_index

def head_np(lst,q_val):
    #lst[0] is chunk text not root
    if(lst[0] == []):
        return 'pneumonoultramicroscopicsilicovolcanoconiosis1'
    idx = get_wh_index(lst[0], q_val)
    # may have "what city" or ["who","the house"]
    head = lst[0][idx:]

    if ("wh" in head[0].lower() and len(head[0]) > 5):
        return lst[1][idx]
    else:
        try:
            return lst[1][idx+1]
        except:
            return lst[1][0]


def get_target_elems(np_list):
    """return chunked_nouns"""
    q_val = get_Wh(doc_question)
    temp = ""
    if(q_val!="none"):
        x = np_list[0]
        inner_index = get_wh_index(x, q_val)

        temp = x[inner_index]
        x.remove(temp)
    return x

def check_NE(doc, question):
    entity_set_p , _ = get_Entities(doc)
    entity_set_q , _ = get_Entities(question)
    if(len(question)>0):
        for i in entity_set_q:
            if i in entity_set_p:
                return True
    return False

def check_NP(doc, question):
    """
    checks to see if some of the NPs from the question are present in the doc
    """
    np_text_doc, np_root_doc = get_NPs(doc)
    np_text_ques, np_root_ques= get_NPs(question)
    for np in np_root_ques:
        # also consider getting lemma
        if np in np_text_doc:
            return True
    return False

def check_POS_Entity(doc,question):
    """checks to see if the q_pronoun's pronoun is present in the doc"""
    q_pronoun = get_Wh(question)
    # get the expected pronoun for the q_pronoun
    wh_pos = map_Wh(q_pronoun)
    text_list, ent_tags = get_Entities(doc)
    for ent in ent_tags:
        if ent in wh_pos:
            return True
    return False

def valid_pos_question(doc, question):
    temp_v = check_NE(doc, question)
    temp_i = check_POS_Entity(doc, question)
    return (temp_v and temp_i)


def dictify(ne_list, text_list):
    lst=zip(ne_list, text_list)
    my_dict = defaultdict(list)
    for k, v in lst:
        my_dict[k].append(v)
    return my_dict

print("--------------NP")

def passes(doc,qdoc):
    x , y = (get_NPs(qdoc), get_Wh(qdoc))
    qhead = head_np(x,y)
    chunk_func = chunk_similarity(doc, qdoc)
    match_func = False
    sim_func = False
    for chunk in doc.noun_chunks:
        chunk = chunk.root.text
        if isMatch(qhead,chunk):
            match_func = True
        if isSim(qhead,chunk):
            sim_func = True

    not_wh_known = True
    #checking if who, when, where
    if (y in question_pronoun_known):
        not_wh_known = valid_pos_question(doc, qdoc)
    return (match_func and chunk_func and not_wh_known)

def open_file(file_path):
    opn = open(file_path.format(0)).read()
    return opn.encode('utf-8')
try:
    path_to_data = sys.argv[1]
except:
    path_to_data = os.path.join(os.getcwd(), "testing.json")

try:
    path_to_test = sys.argv[2]
except:
    path_to_test = os.path.join(os.getcwd(), "final_test_baseline.json")

parsed_json = json.loads(open_file(path_to_data))
count = 0
# answers = []
d = {}
for i in tqdm(range(len(parsed_json["data"]))):
    # for all the paragraphs in data
    for j in range(len(parsed_json["data"][i]["paragraphs"])):
        context = parsed_json["data"][i]["paragraphs"][j]["context"]
        qas = parsed_json["data"][i]["paragraphs"][j]["qas"]

        #for all the questions and answers pertaining to those paragraphs
        for k in range(len(parsed_json["data"][i]["paragraphs"][j]["qas"])):
            if parsed_json["data"][i]["paragraphs"][j]["qas"] != []:
                question = parsed_json["data"][i]["paragraphs"][j]["qas"][k]["question"]
                id_ = parsed_json["data"][i]["paragraphs"][j]["qas"][k]["id"]
                # answer = parsed_json["data"][i]["paragraphs"][j]["qas"][k]["answers"]
                # imposs = parsed_json["data"][i]["paragraphs"][j]["qas"][k]["is_impossible"]

                doc_p = nlp(context)
                doc_q = nlp(question)

                result = passes(doc_p,doc_q)
                # answers.append(result)
                d[id_] = [1 if result is True else 0][0]
print("My program took", time.time() - start_time, "to run")

with open(path_to_test, "w") as outfile:
    json.dump(d, outfile, indent=4)
