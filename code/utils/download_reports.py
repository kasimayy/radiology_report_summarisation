#import wget
import json
import csv
import os, sys

# url = 'http://openi.nlm.nih.gov/retrieve.php?query=Indiana+University+Chest+X-ray+Collection&coll=iu'

# for i in range(1, 248):
#     index = i*30
#     new_set = wget.download(url + '&m=' + str(index+1) + '&n=' + str(index+30))

    
folder = "/vol/medic02/users/ag6516/image_text_modelling/not_needed/data/chestx/text_reports/raw" # or wherever you've moved them all

for filename in os.listdir(folder):
    infilename = os.path.join(folder,filename)
    if not os.path.isfile(infilename): continue
    oldbase = os.path.splitext(filename)
    newname = infilename.replace('.php', '.json')
    output = os.rename(infilename, newname)

mesh_reports = {}

with open('openi.mesh') as fr:
    lines = fr.readlines()

for line in lines:
    linel = line.split('|')
    imagpath = linel[0].lstrip().rstrip().split('/')[-1]
    imagid = 'CXR' + os.path.splitext(imagpath)[0]
    #print imagid
    mesh = linel[1].lstrip().rstrip().lower()
    #print caption
    mesh_ = {}
    mesh_['mesh caption'] = mesh
    mesh_reports[imagid] = mesh_
    #captions.append((imagid, caption))

text_reports = {}
missing_findings = 0
missing_impression = 0
missing_both = 0
imagids = []
for filename in os.listdir(folder):
    fn = os.path.join(folder, filename)
    with open(fn) as data_file:
        data = json.load(data_file)
        list = data['list']
        for idx in range(len(list)):
            imd = list[idx]['imgLarge'].split('/')[-1].split('.')[0]
            if imd not in imagids:
                imagids.append(imd)
                #print imd
                report = list[idx]['abstract'].lower().replace('<p>', ' ').replace('</p>', ' ').replace('<b>', ' ').replace('</b>', ' ').replace('(', ' ').replace(')', ' ').replace('.', ' . ').replace(', ', ' , ').replace('/', ' ').split()
                report_={}
                # print report
                findings_exist = False
                impression_exist = False
                for i, elem in enumerate(report):
                    if 'findings' in elem:
                        findex = i
                        findings_exist = True
                        break
                #print report[findex]
                for i, elem in enumerate(report):
                    if 'impression' in elem:
                        iindex = i
                        impression_exist = True
                        break
                #print report[iindex]
                if findings_exist and impression_exist:
                    findings = report[findex+1:iindex]
                    #print findings
                    impression = report[iindex+1:]
                    #print impression
                    #full_report = {'imagid': imd, 'findings': ' '.join(findings), 'impression': ' '.join(impression)}
                    #text_reports.append([imd, ' '.join(findings) + '.', ' '.join(impression) + '.'])
                    report_['text report'] = [' '.join(findings), ' '.join(impression)]
                    text_reports[imd] = report_
                    #full_report_t = {'imagid': imd, 'findings': findings + ['.'], 'impression': impression + ['.']}
                    #text_reports_t.append(full_report_t)
                elif findings_exist:
                    findings = report[findex+1:]
                    #bad_report = {'imagid': imd, 'findings': ' '.join(findings), 'impression': ' '}
                    #text_reports.append([imd, ' '.join(findings) + '.', ' '])
                    report_['text report'] = [' '.join(findings), ' ']
                    text_reports[imd] = report_
                    #bad_report_t = {'imagid': imd, 'findings': findings + ['.'], 'impression': [' ']}
                    missing_impression+=1
                    #text_reports_t.append(bad_report_t)
                elif impression_exist:
                    impression = report[iindex+1:]
                    #bad_report = {'imagid': imd, 'findings': ' ', 'impression': ' '.join(impression)}
                    #text_reports.append([imd, ' ', ' '.join(impression) + '.'])
                    report_['text report'] = [' ', ' '.join(impression)]
                    text_reports[imd] = report_
                    #bad_report_t = {'imagid': imd, 'findings': [' '], 'impression': impression + ['.']}
                    missing_findings+=1
                    #text_reports_t.append(bad_report_t)
                else:
                    #bad_report = {'imagid': imd, 'findings': [' '], 'impression': [' ']}
                    report_['text report'] = [' ', ' ']
                    missing_both+=1
                #text_reports[imd] = report_

# print missing_findings
# print missing_impression
# print missing_both
print(len(text_reports))

# with open('downloaded_reports.csv', 'w') as csvf:
#     csvw = csv.writer(csvf, delimiter='|')
#     for report in all_reports:
#         csvw.writerow(report)

# with open('unprocessed_reports-tokenized.json', 'w') as fout:
#     json.dump(all_reports_t, fout)

from itertools import chain
from collections import defaultdict
all_reports = []

for k,v in mesh_reports.items():
    if k in text_reports:
        all_reports.append({'imageid': k, 'mesh caption': mesh_reports[k]['mesh caption'], 'text report': text_reports[k]['text report']})

with open('all_reports.json', 'w') as outfile:
    json.dump(all_reports, outfile)
