#!/usr/bin/env python
# coding: utf-8

# # Basic Description
# 
# Current implementation only searches JSON files recursively in the folder, convert these to a dataframe, and processes the texts in paralell. Processing details below.
# 
# * Text is first labeled by language. If english:
# 
# * Text acronyms are expanded. i.e. ADD --> Attention Deficit Disorder. This is done using the acronym expansion module in scispaCy (see their homepage for documentation).
# 
# * Concepts (general NER) in the text are linked to the Unified Medical Language System (UMLS) and canonicalized. The first alias for the entity is appended to the UMLS column.
# 
# * Text is non-destructively lemmatized. No stop words, no deletions of punctuation. For TF-IDF or other algorithms that depend on tokenization, you'll need to run a filter over this column for dimensionality reducation and cleaner text. This mean Sars-Covid-19 stays Sars-Covid-19 as a single token. If you need to match drug names, you can do full-text search on the "sentence" column, or attempt to match to tokens in UMLS, or match NER results in DRUG column.
# 
# * A second pass on NER is run using four NER-specific models from scispaCy. "en_ner_craft_md", "en_ner_jnlpba_md","en_ner_bc5cdr_md","en_ner_bionlp13cg_md". For more information, please see their homepage.
# 
# 
# ## A note on the Extraction class, and section labels
# 
# * The extraction class needs to be edited to read the metadata file and choose files accordingly. Right now, this is at the top of our priority list for tasks in #datasets, and if you can help with this please PM Brandon Eychaner. 
# 
# * Section labels are _messy_. There are more than 250,000 unique section labels in the JSONs alone. I listed the top 1000 section labels by count and took the obvious ones, and mapped them in the "filter_dict" variable to account for the majority of important sections. This is an area of ongoing work. 
# 

# In[17]:


#!pip install googletrans
#!pip install -U scikit-learn

from googletrans import Translator
import pandas as pd 
import os
import numpy as np
import scispacy
import json
import spacy
from tqdm.notebook import tqdm
from scipy.spatial import distance
import ipywidgets as widgets
from scispacy.abbreviation import AbbreviationDetector
from spacy_langdetect import LanguageDetector
# UMLS linking will find concepts in the text, and link them to UMLS. 
from scispacy.umls_linking import UmlsEntityLinker
import time
from spacy.vocab import Vocab
from multiprocessing import Process, Queue, Manager
from multiprocessing.pool import Pool
from functools import partial
import re
import ast


# In[18]:


def translate(text):
    translator=Translator(dest='en')
    translation=translator.translate(str(text)).text
    return translation

# Returns a dictionary object that's easy to parse in pandas. For tables! :D
def extract_tables_from_json(js):
    json_list = []
    # Figures contain useful information. Since NLP doesn't handle images and tables,
    # we can leverage this text data in lieu of visual data.
    for figure in list(js["ref_entries"].keys()):
        json_dict = ["figref", figure, js["ref_entries"][figure]["text"]]
        json_dict.append(json_dict)
    return json_list

def init_filter_dict(): 
    inverse = dict() 
    d = {
        "discussion": ["conclusions","conclusion",'| discussion', "discussion",  'concluding remarks',
                       'discussion and conclusions','conclusion:', 'discussion and conclusion',
                       'conclusions:', 'outcomes', 'conclusions and perspectives', 
                       'conclusions and future perspectives', 'conclusions and future directions'],
        "results": ['executive summary', 'result', 'summary','results','results and discussion','results:',
                    'comment',"findings"],
        "introduction": ['introduction', 'background', 'i. introduction','supporting information','| introduction'],
        "methods": ['methods','method','statistical methods','materials','materials and methods',
                    'data collection','the study','study design','experimental design','objective',
                    'objectives','procedures','data collection and analysis', 'methodology',
                    'material and methods','the model','experimental procedures','main text',],
        "statistics": ['data analysis','statistical analysis', 'analysis','statistical analyses', 
                       'statistics','data','measures'],
        "clinical": ['diagnosis', 'diagnostic features', "differential diagnoses", 'classical signs','prognosis', 'clinical signs', 'pathogenesis',
                     'etiology','differential diagnosis','clinical features', 'case report', 'clinical findings',
                     'clinical presentation'],
        'treatment': ['treatment', 'interventions'],
        "prevention": ['epidemiology','risk factors'],
        "subjects": ['demographics','samples','subjects', 'study population','control','patients', 
                   'participants','patient characteristics'],
        "animals": ['animals','animal models'],
        "abstract": ["abstract", 'a b s t r a c t','author summary'], 
        "review": ['review','literature review','keywords']}
    
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = [key] 
            else: 
                inverse[item].append(key) 
    return inverse

inverted_dict = init_filter_dict()
    
def get_section_name(text):
    if len(text) == 0:
        return(text)
    text = text.lower()
    if text in inverted_dict.keys():
        return(inverted_dict[text][0])
    else:
        if "case" in text or "study" in text: 
            return("methods")
        elif "clinic" in text:
            return("clinical")
        elif "stat" in text:
            return("statistics")
        elif "intro" in text or "backg" in text:
            return("introduction")
        elif "data" in text:
            return("statistics")
        elif "discuss" in text:
            return("discussion")
        elif "patient" in text:
            return("subjects")
        else: 
            return(text)

def init_nlp():
    nlp = spacy.load("en_core_sci_lg", disable=["tagger"])
    nlp.max_length=2000000

    # We also need to detect language, or else we'll be parsing non-english text 
    # as if it were English. 
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    # Add the abbreviation pipe to the spacy pipeline. Only need to run this once.
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)

    # Our linker will look up named entities/concepts in the UMLS graph and normalize
    # the data for us. 
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(linker)
    
    new_vector = nlp(
               """Positive-sense single‐stranded ribonucleic acid virus, subgenus 
                   sarbecovirus of the genus Betacoronavirus. 
                   Also known as severe acute respiratory syndrome coronavirus 2, 
                   also known by 2019 novel coronavirus. It is 
                   contagious in humans and is the cause of the ongoing pandemic of 
                   coronavirus disease. Coronavirus disease 2019 is a zoonotic infectious 
                   disease.""").vector

    vector_data = {"COVID-19": new_vector,
               "2019-nCoV": new_vector,
               "SARS-CoV-2": new_vector}

    vocab = Vocab()
    for word, vector in vector_data.items():
        nlp.vocab.set_vector(word, vector)
    
    return(nlp, linker)
def init_ner():
    models = ["en_ner_craft_md", "en_ner_jnlpba_md","en_ner_bc5cdr_md","en_ner_bionlp13cg_md"]
    nlps = [spacy.load(model) for model in models]
    return(nlps)

def process_metadata(directory):
    rows = []
    print(directory)
    if directory[-1] != "/": 
        directory = directory + "/"
        
    df1 = pd.read_csv(directory + "metadata_old.csv")
    df2 = pd.read_csv(directory + "metadata.csv")
    df = df2[~df2["cord_uid"].isin(df1["cord_uid"])] 
    df.reset_index(drop=True, inplace=True)
    del df1
    del df2
    
    print(df.index)
    df.fillna("~", inplace=True)
    for i in df[df["has_pmc_xml_parse"] == 1].index:
        section = (str(df.iloc[i].full_text_file) + "/") * 2
        pmcid = df.iloc[i].pmcid
        filename = directory + section + "pmc_json/" + pmcid + ".xml.json"
        try: 
            with open(filename) as paperjs:
                jsfile = json.load(paperjs)
        except:
            print("Problem with", df.iloc[i].cord_uid)
            continue

        _id = df.iloc[i]["cord_uid"]
        if "title" in jsfile.keys():
            rows.append(dict(cord_uid=_id, section="title", subsection=0, text=jsfile["title"]))
        else:
            rows.append(dict(cord_uid=_id, section="title", subsection=0, text=df.iloc[i].title))
        if "abstract" in jsfile.keys():
            if len(jsfile["abstract"]) > 1:
                for j in range(len(jsfile["abstract"])):
                    rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["abstract"][j]["text"]))
            else:
                rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["abstract"]))
        elif "abstract" in jsfile["metadata"].keys():
            rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["metadata"]["abstract"]))
        else: 
            rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=df.iloc[i].abstract))

        sections = list(set([k["section"] for k in jsfile["body_text"]]))

        for section in sections: 
            for l in range(len(jsfile["body_text"])):
                if jsfile["body_text"][l]["section"] == section:
                    if section == '':
                        section = "body_text"
                    rows.append(dict(cord_uid=_id, section=section, 
                                     subsection=l, text=jsfile["body_text"][l]["text"]))

        tables = extract_tables_from_json(jsfile)
        for table in tables:
            rows.append(dict(cord_uid=_id, section=table[0], subsection=table[1], text=table[2]))


    for i in df[(df["has_pmc_xml_parse"] == 0) & (df["has_pdf_parse"] == 1)].index:
        section = (str(df.iloc[i].full_text_file) + "/") * 2
        sha = df.iloc[i].sha
        if len(sha.split("; ")) > 1:
            sha = sha.split("; ")[0]
        filename = directory + section + "pdf_json/" + sha + ".json"
        try:
            with open(filename) as paperjs:
                jsfile = json.load(paperjs)
        except:
            print("Problem with", df.iloc[i].cord_uid)

        _id = df.iloc[i]["cord_uid"]
        if "title" in jsfile.keys():
            rows.append(dict(cord_uid=_id, section="title", subsection=0, text=jsfile["title"]))
        else:
            rows.append(dict(cord_uid=_id, section="title", subsection=0, text=df.iloc[i].title))
        if "abstract" in jsfile.keys():
            if len(jsfile["abstract"]) > 1:
                for j in range(len(jsfile["abstract"])):
                    rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["abstract"][j]["text"]))
            else:
                rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["abstract"]))
        elif "abstract" in jsfile["metadata"].keys():
            rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["metadata"]["abstract"]))
        else: 
            rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=df.iloc[i].abstract))

        sections = list(set([k["section"] for k in jsfile["body_text"]]))

        for section in sections: 
            for l in range(len(jsfile["body_text"])):
                if jsfile["body_text"][l]["section"] == section:
                    if section == '':
                        section = "body_text"
                    rows.append(dict(cord_uid=_id, section=section, 
                                     subsection=l, text=jsfile["body_text"][l]["text"]))

        tables = extract_tables_from_json(jsfile)
        for table in tables:
            rows.append(dict(cord_uid=_id, section=table[0], subsection=table[1], text=table[2]))

    for i in df[(df["has_pmc_xml_parse"] == 0) & (df["has_pdf_parse"] == 0)].index:
        section = (str(df.iloc[i].full_text_file) + "/") * 2
        sha = df.iloc[i].sha

        if len(sha.split("; ")) > 1:
            sha = sha.split("; ")[0]
        filename = directory + section + "pdf_json/" + sha + ".json"

        if len(sha) < 2: 
            bad_sha = True
            try:
                with open(directory + section + "pmc_json/" + df.iloc[i]["pmcid"] + ".xml.json") as paperjs:
                    jsfile = json.load(paperjs)
            except:
                pass
        if bad_sha == True:
            bad_sha = False
            continue

        try:
            with open(filename) as paperjs:
                jsfile = json.load(paperjs)
        except:
            print("Problem with ", df.iloc[i].cord_uid)
            continue

        _id = df.iloc[i]["cord_uid"]
        if "title" in jsfile.keys():
            rows.append(dict(cord_uid=_id, section="title", subsection=0, text=jsfile["title"]))
        else:
            rows.append(dict(cord_uid=_id, section="title", subsection=0, text=df.iloc[i].title))
        if "abstract" in jsfile.keys():
            if len(jsfile["abstract"]) > 1:
                for j in range(len(jsfile["abstract"])):
                    rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["abstract"][j]["text"]))
            else:
                rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["abstract"]))
        elif "abstract" in jsfile["metadata"].keys():
            rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=jsfile["metadata"]["abstract"]))
        else: 
            rows.append(dict(cord_uid=_id, section="abstract", subsection=0, text=df.iloc[i].abstract))

        sections = list(set([k["section"] for k in jsfile["body_text"]]))

        for section in sections: 
            for l in range(len(jsfile["body_text"])):
                if jsfile["body_text"][l]["section"] == section:
                    if section == '':
                        section = "body_text"
                    rows.append(dict(cord_uid=_id, section=section, 
                                     subsection=l, text=jsfile["body_text"][l]["text"]))

        tables = extract_tables_from_json(jsfile)
        for table in tables:
            rows.append(dict(cord_uid=_id, section=table[0], subsection=table[1], text=table[2]))
            
    processed_ids = [d["cord_uid"] for d in rows]
    
    for i in df[~df["cord_uid"].isin(processed_ids)].index:
        rows.append(dict(cord_uid=df.iloc[i]["cord_uid"], section="title", subsection=0, text=df.iloc[i]["title"]))
        rows.append(dict(cord_uid=df.iloc[i]["cord_uid"], section="abstract", subsection=0, text=df.iloc[i]["abstract"]))

    return(pd.DataFrame(rows))

def parallelize_dataframe(df, func, n_cores=6, n_parts=400):
    df_split = np.array_split(df, n_parts)
    pool = Pool(n_cores)
    list(tqdm(pool.imap_unordered(func, df_split), total=len(df_split)))
    pool.close()
    pool.join()
    
def noparallelize_dataframe(df, func, n_cores=6, n_parts=400):
    df_split = np.array_split(df, n_parts)
    pool = Pool(n_cores)
    list(tqdm(pool.imap_unordered(func, df_split), total=len(df_split)))
    pool.close()
    pool.join()
                    
def init_list_cols():
    return ['GGP', 'SO', 'TAXON', 'CHEBI', 'GO', 'CL', 'DNA', 'CELL_TYPE', 'CELL_LINE', 'RNA', 'PROTEIN', 
                          'DISEASE', 'CHEMICAL', 'CANCER', 'ORGAN', 'TISSUE', 'ORGANISM', 'CELL', 'AMINO_ACID',
                          'GENE_OR_GENE_PRODUCT', 'SIMPLE_CHEMICAL', 'ANATOMICAL_SYSTEM', 'IMMATERIAL_ANATOMICAL_ENTITY',
                          'MULTI-TISSUE_STRUCTURE', 'DEVELOPING_ANATOMICAL_STRUCTURE', 'ORGANISM_SUBDIVISION',
                          'CELLULAR_COMPONENT', 'PATHOLOGICAL_FORMATION', "lemma", "UMLS","UMLS_ID"]
        
def pipeline(df):
    
    name = df.iloc[0]["cord_uid"] + str(df.iloc[0]["subsection"])+ "0" + ".pickle"
                    
    #if not os.path.exists("df_parts/"):
        #os.mkdir("df_parts/")
        
    #if name in os.listdir("df_parts/"):
    #    return True
    languages = []
    start_chars = []
    end_chars = []
    entities = []
    sentences = []
    lemmas = []
    vectors = []
    subsections = []
    _ids = []
    columns = []
    nlp, linker = init_nlp()
    nlps = init_ner()
    translated = []
    umls_ids = []

    scispacy_ent_types = ['GGP', 'SO', 'TAXON', 'CHEBI', 'GO', 'CL', 'DNA', 'CELL_TYPE', 'CELL_LINE', 'RNA', 'PROTEIN', 
                          'DISEASE', 'CHEMICAL', 'CANCER', 'ORGAN', 'TISSUE', 'ORGANISM', 'CELL', 'AMINO_ACID',
                          'GENE_OR_GENE_PRODUCT', 'SIMPLE_CHEMICAL', 'ANATOMICAL_SYSTEM', 'IMMATERIAL_ANATOMICAL_ENTITY',
                          'MULTI-TISSUE_STRUCTURE', 'DEVELOPING_ANATOMICAL_STRUCTURE', 'ORGANISM_SUBDIVISION',
                          'CELLULAR_COMPONENT', 'PATHOLOGICAL_FORMATION']
    
    for i in tqdm(range(len(df))):
        doc = nlp(str(df.iloc[i]["text"]))
        sents = [sent for sent in doc.sents]

        if len(doc._.abbreviations) > 0 and doc._.language["language"] == "en":
            doc._.abbreviations.sort()
            join_list = []
            start = 0
            for abbrev in doc._.abbreviations:
                join_list.append(str(doc.text[start:abbrev.start_char]))
                if len(abbrev._.long_form) > 5: #Increase length so "a" and "an" don't get un-abbreviated
                    join_list.append(str(abbrev._.long_form))
                else:
                    join_list.append(str(doc.text[abbrev.start_char:abbrev.end_char]))
                start = abbrev.end_char
            # Reassign fixed body text to article in df.
            new_text = "".join(join_list)
            # We have new text. Re-nlp the doc for futher processing!
            doc = nlp(new_text)

        if doc._.language["language"] == "en" and len(doc.text) > 5:
            sents = [sent for sent in doc.sents if len(sent) > 5]
            for sent in sents:
                languages.append(doc._.language["language"])
                sentences.append(sent.text)
                vectors.append(sent.vector)
                translated.append(False)
                subsections.append(df.iloc[i]["subsection"])
                lemmas.append([token.lemma_.lower() for token in sent if not token.is_stop and re.search('[a-zA-Z]', str(token))])
                doc_ents = []
                for ent in sent.ents: 
                    if len(ent._.umls_ents) > 0:
                        poss = linker.umls.cui_to_entity[ent._.umls_ents[0][0]].canonical_name
                        doc_ents.append(poss)
                entities.append(doc_ents)
                umls_ids.append([entity._.umls_ents[0][0] for entity in sent.ents if len(entity._.umls_ents) > 0])
                _ids.append(df.iloc[i]["cord_uid"])
                columns.append(df.iloc[i]["section"])
        else:  
            try: 
                text = translate(df.iloc[i]["text"])
                doc = nlp(str(df.iloc[i]["text"]))
                sents = [sent for sent in doc.sents]

                if len(doc._.abbreviations) > 0:
                    doc._.abbreviations.sort()
                    join_list = []
                    start = 0
                    for abbrev in doc._.abbreviations:
                        join_list.append(str(doc.text[start:abbrev.start_char]))
                        if len(abbrev._.long_form) > 5: #Increase length so "a" and "an" don't get un-abbreviated
                            join_list.append(str(abbrev._.long_form))
                        else:
                            join_list.append(str(doc.text[abbrev.start_char:abbrev.end_char]))
                        start = abbrev.end_char
                    # Reassign fixed body text to article in df.
                    new_text = "".join(join_list)
                    # We have new text. Re-nlp the doc for futher processing!
                    doc = nlp(new_text)

                if len(doc.text) > 5:
                    sents = [sent for sent in doc.sents if len(sent) > 5]
                    for sent in sents:
                        languages.append(doc._.language["language"])
                        sentences.append(sent.text)
                        vectors.append(sent.vector)
                        translated.append(True)
                        subsections.append(df.iloc[i]["subsection"])
                        lemmas.append([token.lemma_ for token in doc if not token.is_stop and re.search('[a-zA-Z]', str(token))])
                        doc_ents = []
                        for ent in sent.ents: 
                            if len(ent._.umls_ents) > 0:
                                poss = linker.umls.cui_to_entity[ent._.umls_ents[0][0]].canonical_name
                                doc_ents.append(poss)
                        umls_ids.append([entity._.umls_ents[0][0] for entity in sent.ents if len(entity._.umls_ents) > 0])
                        entities.append(doc_ents)
                        _ids.append(df.iloc[i]["cord_uid"])
                        columns.append(df.iloc[i]["section"])
                        
            except:
                entities.append("[]")
                translated.append(False)
                subsections.append(df.iloc[i]["subsection"])
                sentences.append(doc.text)
                vectors.append(np.zeros(200))
                lemmas.append("[]")
                _ids.append(df.iloc[i,0])
                umls_ids.append("[]")
                languages.append(doc._.language["language"])
                columns.append(df.iloc[i]["section"])
    
    li1 = _ids
    li2 = subsections
    li3 = [i for i in range(len(entities))]
    
    sentence_id = [str(x) + str(y) + str(z)  for x,y,z in zip(li1,li2,li3)]

    new_df = pd.DataFrame(data={"cord_uid": _ids, "language": languages, "sentence_id": sentence_id,
                                "section": columns, "subsection":subsections, "sentence": sentences,
                                "lemma": lemmas, "UMLS": entities, "UMLS_IDS": umls_ids,
                                "w2vVector": vectors, "translated":translated})
            

    
    for col in scispacy_ent_types:
        new_df[col] = "[]"
    for j in tqdm(new_df.index):
        for nlp in nlps:
            doc = nlp(str(new_df.iloc[j]["sentence"]))
            keys = list(set([ent.label_ for ent in doc.ents]))
            for key in keys:

                # Some entity types are present in the model, but not in the documentation! 
                # In that case, we'll just automatically add it to the df. 
                if key not in scispacy_ent_types:
                    new_df = pd.concat([new_df,pd.DataFrame(columns=[key])])
                    new_df[key] = "[]"

                values = [ent.text for ent in doc.ents if ent.label_ == key]
                new_df.at[j,key] = values

                
    new_df["w2vVector"] = [np.asarray(a=i, dtype="float64") for i in new_df["w2vVector"].to_list()]

    
    new_df.to_pickle("df_parts/" + new_df.iloc[0]["sentence_id"] + ".pickle", compression="gzip")
    return new_df
    #new_df.drop(columns=["w2vVector"]).to_pickle("df_parts/" + new_df.iloc[0]["sentence_id"] + ".ptext", compression="gzip")
    #new_df[["sentence_id","w2vVector"]].to_pickle("df_parts/" + new_df.iloc[0]["sentence_id"] + ".pvec", compression="gzip")


# In[19]:


# Change this to where you have the metadata file. Make sure to untar/unzip all the folders.
directory = "/exchange/CORD-19-research-challenge"

# This method will parse the metadata, add all JSON information according to the highest
# quality source available (Metadata > XML Parse > PDF Parse > leftovers in Metadata)
# The function returns a pandas dataframe. 

df = process_metadata(directory)

# Remove the rows where text was unavailable
df = df[df["text"] != "~"]

# Save the processed data however you like
df.to_pickle("/exchange/v9_dataset.pkl", compression="gzip")


# In[20]:


df = pd.read_pickle("/exchange/v9_dataset.pkl", compression="gzip")
df["text"] = [str(i).replace("((","").replace("))","").replace("(.","").replace(".)","").replace("q q","").replace("\n","") for i in df["text"]]


# In[21]:


len(df)


# In[22]:


mask = (df['text'].str.len() > 10)
df = df.loc[mask]


# In[23]:


len(df)
df.head()


# In[24]:


#parallelize_dataframe(df, pipeline, n_cores=1, n_parts=30)
new_df = pipeline(df)
new_df.head()


# ### Put it all together

# In[25]:


# Change this to whatever version of dataset we're on at this point
version = "v12"


# In[26]:


#df = pd.concat([pd.read_pickle("df_parts/" + f, compression="gzip") for f in os.listdir("df_parts/") if f.endswith(".ptext")])


# In[27]:


# Concatenate and save processed text. 
#df = pd.concat([pd.read_pickle("df_parts/" + f, compression="gzip") for f in os.listdir("df_parts/") if f.endswith(".ptext")])
#df.to_pickle(version + "processedLocalText.pkl", compression="gzip")
jsondf = df.to_json(orient='records')
with open("%s_processedLocalText.json" % version, 'w') as f:
    f.write(jsondf)
#del df

# Concatenate and save processed vectors. 
#df = pd.concat([pd.read_pickle("df_parts/" + f, compression="gzip") for f in os.listdir("df_parts/") if f.endswith(".pvec")])
#df.to_pickle(version + "processedLocalVecs.pkl", compression="gzip")
#del df


# In[28]:


#df = pd.read_pickle("v8processedLocalText.pkl", compression="gzip")


# # Cleanup
# 
# The following code is simply cleanup after the extraction process. First, we'll save the text data in a json file. Next we can save the vector data because it's large, and most people won't be using it. 

# In[29]:


#df = pd.concat([pd.read_pickle("v8processedLocalText.pkl", compression="gzip"), pd.read_pickle("/home/acorn/Downloads/v8processedServerText.pkl", compression="gzip")])


# In[30]:


# VTY
#df.to_pickle("%s_processedText.pkl" % version,compression="gzip")
jsondf = df.to_json(orient='records')
with open("%s_processedText.json" % version, 'w') as f:
    f.write(jsondf)


# In[38]:


string_cols = ['cord_uid', 'language', 'sentence_id', 'section', 'sentence']

int_cols = ["subsection"]

list_cols = [col for col in new_df.columns if col not in string_cols and col not in int_cols]


new_df.fillna("[]", inplace=True)

print(new_df.columns)
for col in new_df.columns:
    #if col in string_cols:
        #new_df[col] = new_df[col].astype("string")
#    if col in list_cols:
#        df[col] = df[col].apply(lambda x: list(x) if not isinstance(x, list) else x)
    if col in int_cols:
        new_df[col] = new_df[col].astype("int")


# In[35]:


new_df['lemma'][0]


# In[39]:


newjsondf = new_df.to_json(orient='records')
with open("%s_sentences.json" % version, 'w') as f:
    f.write(newjsondf)


# In[ ]:




