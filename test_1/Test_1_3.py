import json
import pandas as pd
def main():
    '''
    with open("G://Python//test1//News_Category_Dataset_v2.json", "r+") as f:
        data_list = []
        for line in f:
            jsondata = json.loads(line)
            data_list.append(jsondata)

        count_set = set()
        for jsonItem in data_list:
            count_set.add(jsonItem["category"])
        print(count_set)
        print(len(count_set))
        '''
    #better solution    
    data = pd.read_json('G:/Python/test2/News_Category_Dataset_v2.json', lines=True)
    plotdata = data['category'].value_counts()
    print(plotdata.index,len(plotdata.index))

if __name__ == "__main__":
    main()

#result "41"