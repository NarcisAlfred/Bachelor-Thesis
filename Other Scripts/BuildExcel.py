import os
import xlsxwriter

# The main directory
directory = r"C:\Users\Alfred\Desktop\Importante\Licenta\database\testing"

# The workbook directory
workbook = xlsxwriter.Workbook(directory+"\Data.xlsx")
worksheet = workbook.add_worksheet("Echocardiography Data")

# Add headers to Excel
worksheet.write('A1', 'File Name')
worksheet.write('B1', 'ESV')
worksheet.write('C1', 'Predicted ESV')
worksheet.write('D1', 'EDV')
worksheet.write('E1','Predicted EDV')
worksheet.write('F1', 'EF')
worksheet.write('G1', 'Predicted EF')

print("Started processing files...\n")

# The Excel row iterator
row = 2

for subdir, dirs, files in os.walk(directory):
    print(subdir)
    for file in files:
        if(file == r"Info_2CH.cfg"):
            print(file)
            worksheet.write('A'+str(row),subdir[-11:]+file[-8:-4])
            file = open(subdir + "\%s"%file,'r')
            content = file.read()

            ESV_split = content.split('LVesv: ')[1]
            ESV = ESV_split.split('\n')[0]
            print("ESV: %s"%ESV)
            worksheet.write('B'+str(row),ESV)

            EDV_split = content.split('LVedv: ')[1]
            EDV = EDV_split.split('\n')[0]
            print("EDV: %s"%EDV)
            worksheet.write('D'+str(row),EDV)

            EF = content.split('LVef: ')[1]
            print("EF: %s"%EF)
            worksheet.write('F'+str(row),EF)

            row += 1

workbook.close()