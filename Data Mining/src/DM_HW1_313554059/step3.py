import sys
import time
import os
import pandas as pd
import collections
import numpy as np
import itertools
import math
from optparse import OptionParser

class FPTree:
    def __init__(self, rank_map=None):
        self.root = FPNode(None)
        self.nodes = collections.defaultdict(list)
        self.conditional_items = []
        self.rank_map = rank_map

    def conditional_tree(self, cond_item, min_support_count):
        branches = []
        item_count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                item_count[item] += node.count

        filtered_items = [item for item in item_count if item_count[item] >= min_support_count]
        filtered_items.sort(key=item_count.get)
        updated_rank = {item: i for i, item in enumerate(filtered_items)}

        cond_tree = FPTree(updated_rank)
        for idx, branch in enumerate(branches):
            sorted_branch = sorted(
                [i for i in branch if i in updated_rank],
                key=updated_rank.get,
                reverse=True
            )
            cond_tree.insert_itemset(sorted_branch, self.nodes[cond_item][idx].count)
        cond_tree.conditional_items = self.conditional_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1):
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
            else:
                new_node = FPNode(item, count, node)
                node.children[item] = new_node
                self.nodes[item].append(new_node)
                node = new_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or (
                self.nodes[i][0].children and len(self.nodes[i][0].children) > 1
            ):
                return False
        return True

class FPNode:
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

    def itempath_from_root(self):
        path = []
        node = self
        while node.parent is not None and node.parent.item is not None:
            node = node.parent
            path.append(node.item)
        path.reverse()
        return path

def setup_fptree(df, min_support):
    num_transactions = len(df.index)
    transaction_data = df.values
    item_support = np.sum(df.values == 1, axis=0) / float(num_transactions)
    frequent_items = np.nonzero(item_support >= min_support)[0]

    sorted_indices = item_support[frequent_items].argsort()
    rank_map = {item: i for i, item in enumerate(frequent_items[sorted_indices])}

    # 建立索引到項名稱的映射
    index_to_name = {idx: item for idx, item in enumerate(df.columns)}

    fptree = FPTree(rank_map)
    for i in range(num_transactions):
        itemset = [
            item for item in np.where(transaction_data[i, :])[0] if item in rank_map
        ]
        itemset.sort(key=rank_map.get, reverse=True)
        fptree.insert_itemset(itemset)

    return fptree, rank_map, index_to_name

def generate_itemsets(generator, min_support, index_to_name, num_transactions):
    itemsets = []
    support_values = []
    for support, itemset in generator:
        # 將索引轉換為實際項名稱
        named_itemset = set(index_to_name[item] for item in itemset)
        itemsets.append(named_itemset)
        support_values.append(support / num_transactions)
    result_df = pd.DataFrame({"support": support_values, "itemsets": itemsets})
    return result_df[result_df["support"] >= min_support]

def fpgrowth(df, min_support=0.5):
    fptree, rank_map, index_to_name = setup_fptree(df, min_support)
    num_transactions = len(df.index)
    min_support_count = math.ceil(min_support * num_transactions)  # 支持度換算為出現次數
    generator = fpg_step(fptree, min_support_count)
    return generate_itemsets(generator, min_support, index_to_name, num_transactions)

def fpg_step(tree, min_support_count):
    items = list(tree.nodes.keys())
    if tree.is_path():
        max_size = len(items) + 1
        for size in range(1, max_size):
            for itemset in itertools.combinations(items, size):
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.conditional_items + list(itemset)
    else:
        for item in items:
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.conditional_items + [item]

    if not tree.is_path():
        for item in items:
            cond_tree = tree.conditional_tree(item, min_support_count)
            for support, itemset in fpg_step(cond_tree, min_support_count):
                yield support, itemset

def dataFromFile(filename):
    """讀取資料集"""
    with open(filename, "r") as file_iter:
        for line in file_iter:
            line = line.strip()
            items = line.split()
            items = items[3:]
            yield items

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="dataset名稱", default='datasetA.data'
    )
    optparser.add_option(
        "-s", "--minSupport", dest="minSupport", help="minSupport", default=0.003, type="float"
    )
    optparser.add_option(
        "-p", "--outputPath", dest="output", help="output path", default='./'
    )

    (options, args) = optparser.parse_args()

    if not options.input:
        sys.exit("No dataset filename specified, system will exit")

    data = dataFromFile(options.input)
    min_support = options.minSupport
    output_dir = options.output
    dataset_name = os.path.basename(options.input).split('.')[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Start Task 1： find all frequent itemsets")
    start_time_task1 = time.time()

    transaction_list = list(data)
    df = pd.DataFrame([{item: 1 for item in t} for t in transaction_list]).fillna(0)
    frequent_items_df = fpgrowth(df.astype('bool'), min_support=min_support)
    frequent_items = [
        (frozenset(row['itemsets']), row['support']) for _, row in frequent_items_df.iterrows()
    ]

    output_file = os.path.join(
        output_dir, f'step3_task1_{dataset_name}_{min_support}_result1.txt'
    )
    with open(output_file, 'w') as f:
        frequent_items = sorted(frequent_items, key=lambda x: x[1], reverse=True)
        print(f"number of the frequent itemsets: {len(frequent_items)}")
        for itemset, support in frequent_items:
            support_percent = round(support * 100, 1)
            itemset_str = ','.join(sorted(itemset))
            f.write(f"{support_percent}\t{{{itemset_str}}}\n")

    elapsed_time_task1 = time.time() - start_time_task1
    print(f"Task 1 computation time: {elapsed_time_task1:.4f} s")
