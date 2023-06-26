#import matplotlib.pyplot as plt
from datetime import datetime
import dateutil.parser
import csv
from collections import defaultdict

test = open('test_data.txt', 'r')
text_to_list = []
weight_to_list = []
count = 0

#from txt to list
for i in test:
    text = i.split(",")
    weight_to_list.append(float(text[2]))
    text_to_list.append(text)
    count+= 1

#sorting datetime
text_to_list.sort(key=lambda x: datetime.strptime(x[3], '%d/%m/%Y'))

#reducing list size
weight_to_list.pop(14)

# Standard deviation of list
mean = sum(weight_to_list) / len(weight_to_list)
variance = sum([((x - mean) ** 2) for x in weight_to_list]) / len(weight_to_list)
res = variance ** 0.5
##print("Standard deviation of sample is : " + str(res))

for i in range(0,text_to_list.__len__()):
    if i > 0:
        normal = True
        prev = i-1
        diff = (float(text_to_list[i][2]) - float(text_to_list[prev][2]))
        if diff > res or diff < -res:
            normal = False
        text_to_list[i].append(diff)
        text_to_list[i].append(normal)
        ##print("Date: ", text_to_list[i][3], "||", text_to_list[prev][2], ", ", text_to_list[i][2], "| diff = ", diff, "|" , normal)
    else:
        text_to_list[i].append(0)
        text_to_list[i].append(0)

#reprinting the whole thing
'''
with open(r'test_file_1.txt', 'w') as fp:
    for item in text_to_list:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
    '''


#Testing with raw unedited

test2 = open('test_data_raw.txt','r')
ttl = []
wtl = []
count_ = 0

for i in test2:
    if count_ > 0:
        text = i.split('    ')
        wtl.append(float(text[2]))
        date = dateutil.parser.parse(text[3])
        text[3] = date
        ttl.append(text)
    count_ += 1

ttl.sort(key=lambda x: x[3])

# Standard deviation of list
mean = sum(wtl) / len(wtl)
variance = sum([((x - mean) ** 2) for x in wtl]) / len(wtl)
res = variance ** 0.5
##print("Standard deviation of sample is : " + str(res))

for i in range(0,ttl.__len__()):
    if i > 0:
        normal = True
        prev = i-1
        diff = (float(ttl[i][2]) - float(ttl[prev][2]))
        if diff > res or diff < -res:
            normal = False
        ttl[i].append(diff)
        ttl[i].append(normal)
        ##print("Date: ", ttl[i][3], "||", ttl[prev][2], ", ", ttl[i][2], "\t| diff = ", diff, "\t|" , normal)
    else:
        ttl[i].append(0)
        ttl[i].append(0)



########################################################################################################################



file = open("Belmont_cows_WOW.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
ttl = []
count = 0

for i in data:
    if count > 0:
        date = dateutil.parser.parse(i[3])
        i[3] = date
        ttl.append(i)
    count+=1

groups = defaultdict(list)    

for obj in ttl:
    groups[obj[1]].append(obj)

new_list = groups.values()
total_res = 0

for i in new_list:
    #sorting dates
    i.sort(key = lambda x: x[3])
    wtl = []
    
    for j in i:
        wtl.append(float(j[2]))
    
    # Standard deviation of list
    mean = sum(wtl) / len(wtl)
    variance = sum([((x - mean) ** 2) for x in wtl]) / len(wtl)
    res = variance ** 0.5
    print("Standard deviation of sample", i[0][1] ," is : " + str(res))
    total_res+=res

    for j in range(0,len(i)):
        if j > 0:
            normal = True
            prev = j-1
            diff = (float(i[j][2]) - float(i[prev][2]))
            if diff > res or diff < -res:
                normal = False
            i[j].append(diff)
            i[j].append(normal)
            print("Date: ", i[j][3], "||", i[prev][2], ", ", i[j][2], "\t| diff = ", diff, "\t|" , normal)
        else:
            i[j].append(0)
            i[j].append("True")

print("Average Standard Deviation is ", total_res/len(new_list), "| Total Unique CattleIDs: ", len(new_list))

count = 0

with open(r'allcattle.csv', 'w') as fp:
    fp.write("num,RFID,Weight,Date,Location,cattle_id,prev_diff,within_standard_dev,padding_remove\n")
    for item in new_list:
        for i in item:
            date = i[3].strftime("%m/%d/%Y")
            i[3] = date
            count += 1
            i[0] = count
            for j in i:
                fp.write("%s" % j)
                fp.write(",")
            # write each item on a new line
            fp.write("\n")          
    print('Done')
print(count)

#Writing into individual files for each cattle ID

'''num_of_unique = 0
current = None
for item in new_list:
    filename = item[0][1]
    with open('individualrecords/%s.csv' % filename, 'w') as fp:
        fp.write("num,RFID,Weight,Date,Location,cattle_id,prev_diff,within_standard_dev,padding_remove\n")
        numcount = 1
        for i in item:
            i[0] = numcount
            date = i[3].strftime("%m/%d/%Y")
            i[3] = date
            numcount+=1
            for j in i:
                fp.write("%s" % j)
                fp.write(",")
            # write each item on a new line
            fp.write("\n")
    num_of_unique+=1
print(num_of_unique)
'''