from sklearn.metrics import accuracy_score

def read_col(fname, col, convert=int, sep=None):
    with open(fname) as data:
         return [convert(line.split(sep=sep)[col]) for line in data]
def accr(test,truth):
    truth_list = read_col('G:/Python/test_truth.txt',truth)
    test_list = read_col('G:/Python/test_truth.txt',test)
    accr_scor=accuracy_score(truth_list, test_list)
    print(accr_scor)

if __name__ == "__main__":
    accr(1,0)