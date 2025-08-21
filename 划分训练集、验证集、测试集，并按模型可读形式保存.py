import pandas as pd
import random

# 使用 openpyxl 引擎来读取 .xlsx 文件
df = pd.read_excel(r'D:\年报大模型\bert\gogo\待检测句典.xlsx', sheet_name='Sheet1', dtype={'sentences': str, 'sign': str}, engine='openpyxl')

# 自动获取数据总数
total = len(df)

# 按比例划分训练集、验证集和测试集
L1 = random.sample(range(total), int(total * 0.8))  # 训练集
lastList1 = [x for x in range(total) if x not in L1]
L2 = random.sample(lastList1, int(total * 0.1))  # 验证集
L3 = [x for x in range(total) if x not in L1 and x not in L2]  # 测试集

train = ''
test = ''
dev = ''

# 遍历每一行，按句子,\t,标记的格式保存到txt
for i in range(total):
    if i in L1:
        train = train + df['sentences'][i] + '\t' + str(df['sign'][i]) + '\n'
    elif i in L2:
        test = test + df['sentences'][i] + '\t' + str(df['sign'][i]) + '\n'
    else:
        dev = dev + df['sentences'][i] + '\t' + str(df['sign'][i]) + '\n'

# 将训练集、验证集和测试集保存为txt文件
with open(r'D:\年报大模型\输出\train.txt', 'w', encoding='utf-8') as w:
    w.write(train)
with open(r'D:\年报大模型\输出\test.txt', 'w', encoding='utf-8') as w:
    w.write(test)
with open(r'D:\年报大模型\输出\dev.txt', 'w', encoding='utf-8') as w:
    w.write(dev)
