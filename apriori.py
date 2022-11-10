
from apyori import apriori

# store_data = pd.read_csv('data/store_data.csv', header=None).fillna('')
# store_data.head()
records = []
#读取csv文件，将每一行按逗号分割存成一个List，然后整体存储成List,类似这种形式List[[],[],[]]
with open('store_data.csv', "r", newline='') as f:
    import csv
    reader = csv.reader(f)
    for line in reader:
        records.append(line)
#使用Apriori关联规则分析，设置最小支持度，置信度等
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
#将结果转换成List
association_results = list(association_rules)
#打印首行，可以看看结果是怎样的
print(association_results[0])
# RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, ordered_statistics[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)])
#使用for循环，查看符合条件的关联规则，并打印出来
for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

