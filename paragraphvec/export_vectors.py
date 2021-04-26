import csv
import re
from os.path import join

import fire
import torch

from paragraphvec.data import load_dataset
from paragraphvec.models import DM, DBOW
from paragraphvec.utils import DATA_DIR, MODELS_DIR

from simpleneighbors import SimpleNeighbors
from arango import ArangoClient


def start(data_file_name, model_file_name):
    """Saves trained paragraph vectors to a csv file in the *data* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory that was used during training.

    model_file_name: str
        Name of a file in the *models* directory (a model trained on
        the *data_file_name* dataset).
    """
    dataset = load_dataset(data_file_name)

    vec_dim = int(re.search('_vecdim\.(\d+)_', model_file_name).group(1))

    model = _load_model(
        model_file_name,
        vec_dim,
        num_docs=len(dataset),
        num_words=len(dataset.fields['text'].vocab) - 1)
    
    #_write_to_file(data_file_name, model_file_name, model, vec_dim)
    db = connect_db("nebula_dev")
    index = _write_to_annoy(db, model, vec_dim)
    input_loop(index)



def _load_model(model_file_name, vec_dim, num_docs, num_words):
    model_ver = re.search('_model\.(dm|dbow)', model_file_name).group(1)
    if model_ver is None:
        raise ValueError("Model file name contains an invalid"
                         "version of the model")

    model_file_path = join(MODELS_DIR, model_file_name)

    try:
        checkpoint = torch.load(model_file_path)
    except AssertionError:
        checkpoint = torch.load(
            model_file_path,
            map_location=lambda storage, location: storage)

    if model_ver == 'dbow':
        model = DBOW(vec_dim, num_docs, num_words)
    else:
        model = DM(vec_dim, num_docs, num_words)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def _write_to_file(data_file_name, model_file_name, model, vec_dim):
    result_lines = []

    with open(join(DATA_DIR, data_file_name)) as f:
        reader = csv.reader(f)

        for i, line in enumerate(reader):
            # skip text
            result_line = line[1:]
            if i == 0:
                # header line
                result_line += ["d{:d}".format(x) for x in range(vec_dim)]
            else:
                vector = model.get_paragraph_vector(i - 1)
                result_line += [str(x) for x in vector]

            result_lines.append(result_line)

    result_file_name = model_file_name[:-7] + 'csv'

    with open(join(DATA_DIR, result_file_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_lines)

def _write_to_annoy(db, model, vec_dim):
    query = 'FOR doc IN Stories RETURN doc'  
    stories = []
    embedding_dimensions = vec_dim
    num_index_trees = 512
    single_index = SimpleNeighbors(embedding_dimensions) 
    cursor = db.aql.execute(
            query
        )
    for i, data in enumerate(cursor):
        key = data['movie_id']
        vector = model.get_paragraph_vector(i - 1)
        single_index.add_one(key, vector)
    single_index.build(n=num_index_trees) 
    return(single_index)

def input_loop(index):
    while (True):
            _key = input("Enter movie id: ")
            #Key -> movie id (Movies/1111111), Top -> number of similar movies
            sims = index.neighbors(_key, n=10)
            if sims:
                print("----------------------") 
                print("Top 10 Sims for Movie: " + _key)
                for sim in sims:
                    print(sim)
                print("----------------------")  
            else:
                print("Please rebuild index")

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

if __name__ == '__main__':
    fire.Fire()
