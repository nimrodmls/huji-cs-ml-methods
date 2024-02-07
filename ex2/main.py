import helpers

def read_data(filename):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """
    raw_data, _ = helpers.read_data_demo(filename)

    # The first two columns are the features (Longtitude & Latitude)
    # and the third column is the class label.
    dataset = raw_data[:,:2]
    classes = raw_data[:,2:].flatten()

    return dataset, classes

if __name__ == "__main__":
    train_dataset, train_classes = read_data('train.csv')
    test_dataset, test_classes = read_data('test.csv')

    print(helpers.knn_examples(
        train_dataset, train_classes, test_dataset, test_classes))