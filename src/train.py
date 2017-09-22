# get dataset
# tokenize
# pad sequences
#
# get embeddings
# generate random negative examples on the fly
# embed training data
# pass through model
# get CE loss between negative examples and truth

from data import Dataset

DATA_DIR = '/home/mnoukhov/common-sense-prediction/data'

def main(data_dir):
    dataset = Dataset(data_dir)
    embeddings = dataset.embeddings
    data_stream = dataset.data_stream()

    import pdb;pdb.set_trace()
    for data in data_stream.get_epoch_iterator():
        print(data)


if __name__ == '__main__':
    main(DATA_DIR)
