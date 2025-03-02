import sys
import os
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
from multiprocessing import Pool

def subsets(arr):
    """返回 arr 的非空子集"""
    return chain(*[combinations(arr, i + 1) for i in range(len(arr))])

def count_support(args):
    #獨立出來的支持度計算函數
    item, transactionList = args
    count = sum(1 for transaction in transactionList if item.issubset(transaction))
    return (item, count)

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """计算 itemSet 中各项的支持度，返回满足最小支持度的项集"""
    _itemSet = set()
    num_transactions = len(transactionList)
    
    # 准备多处理的参数
    pool = Pool()
    args = [(item, transactionList) for item in itemSet]
    
    # 使用多处理计算支持度
    results = pool.map(count_support, args)
    pool.close()
    pool.join()
    
    for item, count in results:
        freqSet[item] += count
        support = float(count) / num_transactions
        if support >= minSupport:
            _itemSet.add(item)
    
    return _itemSet

# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length and len(i.intersection(j)) == length - 2])

def getItemSetTransactionList(data_iterator):
#從dataset中取出item作為1-項集存入itemSet
 #並生成由每一筆交易之frozenset組成的transactionList
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record) #取出data中每行ITEMSET並轉為frozenset
        transactionList.append(transaction) #將frozenset存入transactionlist
        for item in transaction:
            itemSet.add(frozenset([item])) #將每行之每個item作為1-項集存入itemSet
    return itemSet, transactionList

def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    stats = []  # 新增 stats 列表以記錄每次迭代的候選數據

    # 處理第一輪 (1-項集) 並記錄候選數據
    num_candidates_before = len(itemSet) # 原始的 1-itemset 候選集數量
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    num_candidates_after = len(oneCSet) # prunning後的 1-itemset 候選集數量
    stats.append((1, num_candidates_before, num_candidates_after)) # 紀錄第一輪的迭代狀況（result2）中需要用到  
    largeSet[1] = oneCSet

    currentLSet = oneCSet
    k = 2

    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentCSet = joinSet(currentLSet, k)
        num_candidates_before = len(currentCSet)
        currentLSet = returnItemsWithMinSupport(currentCSet, transactionList, minSupport, freqSet)
        num_candidates_after = len(currentLSet)
        stats.append((k, num_candidates_before, num_candidates_after))
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)
    items = []
    for key in sorted(largeSet.keys()):
        value = largeSet[key]
        items.extend([(tuple(item), getSupport(item)) for item in value])

    return items, stats, transactionList

def findClosedItemsets(items):
    """計算Closed freqeunt itemset"""
    #宣告supportData字典並且value為每frozenset(key)的support
    supportData = dict()
    for itemset, support in items:
        supportData[frozenset(itemset)] = support
    closedItemsets = [] #存儲閉鎖頻繁集
    allFrequentItemsets = list(supportData.keys()) #將頻繁集轉換為列表方便檢查是否為閉鎖頻繁集
    for itemset in allFrequentItemsets:
        isClosed = True
        itemsetSupport = supportData[itemset] #取得該行itemset的support(value)
        for otherItemset in allFrequentItemsets: 
            if len(otherItemset) > len(itemset):
                #otherItemset是比較大的那個
                if itemset.issubset(otherItemset):
                    #當前itemset是否為otherItemset之子集
                    if supportData[otherItemset] == itemsetSupport:
                        #support相同而且比較大，代表他不是閉鎖頻繁集，改false並退出
                        isClosed = False
                        break
        if isClosed:
            #若沒有比他大的itemset他就是閉鎖頻繁集，加入closedItemsets
            closedItemsets.append((tuple(itemset), itemsetSupport))

    return closedItemsets

def printResults(items, outputFile):
    with open(outputFile, 'w') as f:
        items = sorted(items, key=lambda x: x[1], reverse=True)
        print(f"total number of frequent itemset: {len(items)}")
        for itemset, support in items:
            support_percent = round(support * 100, 1)
            itemset_str = ','.join(sorted(itemset))
            f.write(f"{support_percent}\t{{{itemset_str}}}\n")

def printStats(stats, outputFile):
    total_itemsets = sum([after for _, _, after in stats])
    with open(outputFile, 'w') as f:
        f.write(f"{total_itemsets}\n")
        for stat in stats:
            iteration, before, after = stat
            f.write(f"{iteration}\t{before}\t{after}\n")

def dataFromFile(fname):
    """讀取資料集"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip()
            items = line.split(" ")
            items = items[3:]  #移除前三个字段
            yield frozenset(items) #frozenset為不可變動之集合

if __name__ == "__main__":
    import time

    optparser = OptionParser()
    optparser.add_option("-f", "--inputFile", dest="input", help="dataset名稱", default='datasetA.data')
    optparser.add_option("-s", "--minSupport", dest="minS", help="minSupport", default=0.003, type="float")
    optparser.add_option("-p", "--outputPath", dest="outputPath", help="output path", default='./')

    (options, args) = optparser.parse_args()
    data_list = list(dataFromFile(options.input)) 

    minSupport = options.minS
    script_base_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    input_filename = options.input
    input_base_name = os.path.basename(input_filename)
    if input_base_name.endswith('.data'):
        dataset_name = input_base_name[:-5]
    else:
        dataset_name = input_base_name
    min_support_str = str(minSupport).rstrip('0').rstrip('.')
    output_path = options.outputPath if options.outputPath else "."
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file1 = f"{script_base_name}_task1_{dataset_name}_{min_support_str}_result1.txt"
    output_file2 = f"{script_base_name}_task1_{dataset_name}_{min_support_str}_result2.txt"
    output_file3 = f"{script_base_name}_task2_{dataset_name}_{min_support_str}_result1.txt"
    output_file1 = os.path.join(output_path, output_file1)
    output_file2 = os.path.join(output_path, output_file2)
    output_file3 = os.path.join(output_path, output_file3)

    # -------------------- Task 1 --------------------
    print("Running Task1:")
    start_time_task1 = time.time()
    items_task1, stats_task1, _ = runApriori(data_list, minSupport)
    printResults(items_task1, output_file1)
    printStats(stats_task1, output_file2)
    elapsed_time_task1 = time.time() - start_time_task1
    print(f"Task1 time: {elapsed_time_task1:.4f} seconds")
    # -------------------- Task 2 --------------------
    print("\nRunning Task2:")
    start_time_task2 = time.time()
    # 使用task1中的frequent itemset進一步計算closed frequent itemset
    closed_items = findClosedItemsets(items_task1)
    # 输出 Task 2 结果
    with open(output_file3, 'w') as f:
        total_closed_itemsets = len(closed_items)
        f.write(f"{total_closed_itemsets}\n")
        closed_items = sorted(closed_items, key=lambda x: x[1], reverse=True)
        print(f"total number of frequent closed itemset: {total_closed_itemsets}")
        for itemset, support in closed_items:
            support_percent = round(support * 100, 1)
            itemset_str = ','.join(sorted(itemset))
            f.write(f"{support_percent}\t{{{itemset_str}}}\n")

    elapsed_time_task2 = time.time() - start_time_task2
    print(f"Task2 time: {elapsed_time_task2:.4f} seconds")

    # 计算时间比率 (Task2 / Task1 * 100%)
    time_ratio = (elapsed_time_task2 / elapsed_time_task1) * 100
    print(f"\n time ratio of (Task2 / Task1 * 100%): {time_ratio:.2f}%")
