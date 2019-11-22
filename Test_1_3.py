import json
def main():
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

if __name__ == "__main__":
    main()